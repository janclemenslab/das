from copy import deepcopy
from collections.abc import Mapping, Sequence

import lightning as L
import torch
import torchmetrics
from torch import nn, optim

from .decoders import (
    DecoderConfig,
    build_decoder,
    normalize_decoder_config,
    serialize_decoder_config,
)
from .encoders import EncoderConfig, build_encoder, normalize_encoder_config, serialize_encoder_config
from .frontends import (
    FrontendConfig,
    SincFrontendConfig,
    build_frontend,
    frontend_hop_length_samples,
    normalize_frontend_config,
    serialize_frontend_config,
)


def syllcount_loss(output_labels: torch.Tensor, target_labels: torch.Tensor) -> torch.Tensor:
    if output_labels.ndim == 1:
        output_labels = output_labels.unsqueeze(0)
        target_labels = target_labels.unsqueeze(0)

    if output_labels.shape[1] < 2:
        return output_labels.new_tensor(0.0)

    nb_syll_true = torch.sum(torch.diff(target_labels, dim=1) >= 1, dim=1).float()
    nb_syll_pred = torch.sum(torch.diff(output_labels, dim=1) >= 1, dim=1).float()
    return torch.mean(torch.square(nb_syll_true - nb_syll_pred))


class DASModel(L.LightningModule):
    def __init__(
        self,
        num_classes: int,
        sr: float,
        *,
        class_names: Sequence[str] | None = None,
        class_types: Sequence[str] | None = None,
        frontend: FrontendConfig | Mapping[str, object],
        encoder: EncoderConfig | Mapping[str, object],
        decoder: DecoderConfig | Mapping[str, object],
        cross_entropy_weight: float,
        learning_rate: float,
        positive_class_weight: float = 1.0,
        boundary_weight: float = 1.0,
        boundary_width_ms: float = 20.0,
        reduce_lr: bool = True,
        reduce_lr_patience: int = 5,
        reduce_lr_factor: float = 0.1,
        reduce_lr_min: float = 1e-8,
        reduce_lr_monitor: str = "train_loss",
        num_time_steps: int | None = None,
        chunk_stride: int | None = None,
        checkpoint_metadata: Mapping[str, object] | None = None,
    ):
        super().__init__()
        frontend_config = normalize_frontend_config(frontend)
        encoder_config = normalize_encoder_config(encoder)
        decoder_config = normalize_decoder_config(decoder)
        self.checkpoint_metadata = {} if checkpoint_metadata is None else deepcopy(dict(checkpoint_metadata))

        self.save_hyperparameters(
            {
                "num_classes": int(num_classes),
                "sr": float(sr),
                "class_names": None if class_names is None else list(class_names),
                "class_types": None if class_types is None else list(class_types),
                "frontend": serialize_frontend_config(frontend_config, include_raw_num_channels=True),
                "encoder": serialize_encoder_config(encoder_config),
                "decoder": serialize_decoder_config(decoder_config),
                "cross_entropy_weight": float(cross_entropy_weight),
                "positive_class_weight": float(positive_class_weight),
                "boundary_weight": float(boundary_weight),
                "boundary_width_ms": float(boundary_width_ms),
                "learning_rate": float(learning_rate),
                "reduce_lr": bool(reduce_lr),
                "reduce_lr_patience": int(reduce_lr_patience),
                "reduce_lr_factor": float(reduce_lr_factor),
                "reduce_lr_min": float(reduce_lr_min),
                "reduce_lr_monitor": str(reduce_lr_monitor),
                "num_time_steps": None if num_time_steps is None else int(num_time_steps),
                "chunk_stride": None if chunk_stride is None else int(chunk_stride),
            }
        )

        self.num_classes = int(num_classes)
        self.learning_rate = float(learning_rate)
        self.reduce_lr = bool(reduce_lr)
        self.reduce_lr_patience = int(reduce_lr_patience)
        self.reduce_lr_factor = float(reduce_lr_factor)
        self.reduce_lr_min = float(reduce_lr_min)
        self.reduce_lr_monitor = str(reduce_lr_monitor)
        self.cross_entropy_weight = float(cross_entropy_weight)
        self.boundary_weight = float(boundary_weight)
        self.class_names = None if class_names is None else list(class_names)
        self.class_types = None if class_types is None else list(class_types)
        self.frontend_config = frontend_config
        self.encoder_config = encoder_config
        self.decoder_config = decoder_config
        self.frontend = build_frontend(frontend_config, sr=sr)
        self.encoder = build_encoder(encoder_config, input_dim=self.frontend.output_dim, sr=sr)
        self.decoder = build_decoder(
            decoder_config,
            input_dim=self.encoder.output_dim,
            num_classes=self.num_classes,
        )

        hop_samples = frontend_hop_length_samples(frontend_config, sr=sr)
        self.boundary_width_frames = max(0, int(round(float(boundary_width_ms) * float(sr) / 1000.0 / hop_samples)))

        class_weights = (
            None
            if float(positive_class_weight) == 1.0
            else torch.tensor([1.0] + [float(positive_class_weight)] * (self.num_classes - 1))
        )
        self.criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            reduction="mean",
            ignore_index=-100,
        )
        self._freeze_encoder = False

    def on_save_checkpoint(self, checkpoint: dict[str, object]) -> None:
        if self.checkpoint_metadata:
            checkpoint["das"] = deepcopy(self.checkpoint_metadata)

    def freeze_encoder(self) -> None:
        self.encoder.requires_grad_(False)
        self.encoder.eval()
        self._freeze_encoder = True

    def train(self, mode: bool = True):
        result = super().train(mode)
        if mode and self._freeze_encoder:
            self.encoder.eval()
        return result

    def forward(
        self,
        inputs: torch.Tensor,
        input_lengths: torch.Tensor | None = None,
    ):
        input_lengths = self._normalize_input_lengths(inputs, input_lengths)
        channel_shape = None
        if inputs.ndim == 3 and isinstance(self.frontend_config, SincFrontendConfig):
            batch_size, num_channels, num_samples = inputs.shape
            inputs = inputs.reshape(batch_size * num_channels, num_samples)
            input_lengths = input_lengths.repeat_interleave(num_channels)
            channel_shape = (batch_size, num_channels)
        features, feature_lengths = self.frontend(inputs, input_lengths)
        if channel_shape is not None:
            batch_size, num_channels = channel_shape
            features = features.reshape(batch_size, num_channels, *features.shape[1:]).amax(dim=1)
            feature_lengths = feature_lengths.reshape(batch_size, num_channels)[:, 0]
        encoded, output_lengths = self.encoder(features, feature_lengths)
        logits = self.decoder(encoded)
        return logits, output_lengths

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        del batch_idx
        metrics = self._shared_step(batch, include_metrics=False)
        self._log_dict(
            {
                "train_loss": metrics["loss"],
                "train_crossentropy": metrics["loss_xent"],
                "train_syllcount": metrics["loss_nbsyll"],
            }
        )
        return metrics["loss"]

    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        del batch_idx
        metrics = self._shared_step(batch, include_metrics=False)
        values = {"val_loss": metrics["loss"], "lr": self._current_lr()}
        values.update(
            {
                "val_crossentropy": metrics["loss_xent"],
                "val_syllcount": metrics["loss_nbsyll"],
            }
        )
        self._log_dict(values, prog_bar=True)
        return metrics["loss"]

    def test_step(self, batch, batch_idx: int) -> torch.Tensor:
        del batch_idx
        metrics = self._shared_step(batch, include_metrics=True)
        values = {"test_loss": metrics["loss"]}
        values.update(
            {
                "test_acc": metrics["acc"],
                "test_f1": metrics["f1"],
                "test_precision": metrics["precision"],
                "test_recall": metrics["recall"],
            }
        )
        self._log_dict(values, prog_bar=True)
        return metrics["loss"]

    def predict_step(self, batch, batch_idx: int):
        del batch_idx
        inputs, input_lengths, _, _ = self._unpack_batch(batch)
        outputs, output_lengths = self.forward(inputs, input_lengths)
        return outputs, output_lengths

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        if not self.reduce_lr:
            return optimizer
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.reduce_lr_factor,
            patience=self.reduce_lr_patience,
            cooldown=2,
            min_lr=self.reduce_lr_min,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": self.reduce_lr_monitor,
            },
        }

    def _shared_step(self, batch, include_metrics: bool) -> dict[str, torch.Tensor]:
        inputs, input_lengths, targets, _ = self._unpack_batch(batch)
        logits, _ = self.forward(inputs, input_lengths)
        logits, target_indices = self._align_logits_and_targets(logits, targets)

        flat_logits = logits.reshape(-1, self.num_classes)
        flat_targets = target_indices.reshape(-1)
        predicted_labels = torch.argmax(logits, dim=-1)

        loss_xent = self._dense_cross_entropy(flat_logits, flat_targets, target_indices)
        loss_nbsyll = syllcount_loss(predicted_labels, target_indices)
        loss = self.cross_entropy_weight * loss_xent + (1 - self.cross_entropy_weight) * loss_nbsyll

        results = {
            "loss": loss,
            "loss_xent": loss_xent,
            "loss_nbsyll": loss_nbsyll,
        }
        if include_metrics:
            results.update(self._compute_metrics(predicted_labels.reshape(-1), flat_targets))
        return results

    def _dense_cross_entropy(
        self,
        flat_logits: torch.Tensor,
        flat_targets: torch.Tensor,
        target_indices: torch.Tensor,
    ) -> torch.Tensor:
        if self.boundary_weight == 1.0 or self.boundary_width_frames == 0:
            return self.criterion(flat_logits, flat_targets)

        transitions = target_indices[:, 1:] != target_indices[:, :-1]
        boundaries = torch.zeros_like(target_indices, dtype=torch.bool)
        boundaries[:, 1:] |= transitions
        boundaries[:, :-1] |= transitions
        width = self.boundary_width_frames
        boundaries = nn.functional.max_pool1d(
            boundaries.float().unsqueeze(1), kernel_size=2 * width + 1, stride=1, padding=width
        ).squeeze(1)
        weights = 1.0 + (self.boundary_weight - 1.0) * boundaries
        losses = nn.functional.cross_entropy(
            flat_logits,
            flat_targets,
            weight=self.criterion.weight,
            reduction="none",
        ).reshape_as(target_indices)
        return (losses * weights).sum() / weights.sum()

    def _normalize_input_lengths(self, inputs: torch.Tensor, input_lengths: torch.Tensor | None) -> torch.Tensor:
        if input_lengths is None:
            return torch.full(
                (inputs.shape[0],),
                fill_value=inputs.shape[-1],
                device=inputs.device,
                dtype=torch.long,
            )

        return torch.as_tensor(input_lengths, device=inputs.device, dtype=torch.long)

    def _align_logits_and_targets(self, logits: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if targets.ndim == logits.ndim:
            target_indices = torch.argmax(targets, dim=-1)
        else:
            target_indices = targets.long()

        time_steps = min(logits.shape[1], target_indices.shape[1])
        return logits[:, :time_steps], target_indices[:, :time_steps]

    def _unpack_batch(self, batch):
        if len(batch) == 4:
            return batch
        if len(batch) == 2:
            inputs, input_lengths = batch
            return inputs, input_lengths, None, None
        return batch

    def _log_dict(self, values: dict[str, torch.Tensor | float], prog_bar: bool = False) -> None:
        if getattr(self, "_trainer", None) is None:
            return
        self.log_dict(values, prog_bar=prog_bar)

    def _current_lr(self) -> float:
        trainer = getattr(self, "_trainer", None)
        if trainer is None or not trainer.optimizers:
            return self.learning_rate
        return trainer.optimizers[0].param_groups[0]["lr"]

    def _compute_metrics(self, predicted_labels: torch.Tensor, target_labels: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "acc": torchmetrics.functional.classification.multiclass_accuracy(
                predicted_labels,
                target_labels,
                num_classes=self.num_classes,
                average="macro",
            ),
            "f1": torchmetrics.functional.classification.multiclass_f1_score(
                predicted_labels,
                target_labels,
                num_classes=self.num_classes,
                average="macro",
            ),
            "precision": torchmetrics.functional.classification.multiclass_precision(
                predicted_labels,
                target_labels,
                num_classes=self.num_classes,
                average="macro",
            ),
            "recall": torchmetrics.functional.classification.multiclass_recall(
                predicted_labels,
                target_labels,
                num_classes=self.num_classes,
                average="macro",
            ),
        }
