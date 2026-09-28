import os
from typing import Any, Dict, Literal, Optional, Sequence

import lightning.pytorch as pl
from lightning.pytorch.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)
from lightning.pytorch.loggers import CSVLogger
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from .datautils import (
    VocalSegDataset,
    build_vocalseg_clip_records,
    determine_default_config,
    get_audio_and_label_paths,
    get_cluster_codebook,
    read_label,
    resolve_training_data_dir,
)
from ..data.audio_dir import _normalize_include_labels, _validate_include_labels
from ..progress import TrainingLogProgress
from .model import checkpoint_payload, load_model


class WhisperSegLightningModule(pl.LightningModule):
    def __init__(
        self,
        model=None,
        *,
        learning_rate: float,
        weight_decay: float,
        linear_lr_schedule: bool,
        warmup_steps: int,
        reduce_lr: bool,
        reduce_lr_patience: int,
        reduce_lr_factor: float,
        reduce_lr_min: float,
        lr_monitor: str,
        total_training_steps: Optional[int],
        tokenizer=None,
        model_init_config: Optional[Dict[str, Any]] = None,
        checkpoint_metadata: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.model_init_config = model_init_config
        self.checkpoint_metadata = {} if checkpoint_metadata is None else dict(checkpoint_metadata)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.linear_lr_schedule = linear_lr_schedule
        self.warmup_steps = warmup_steps
        self.reduce_lr = reduce_lr
        self.reduce_lr_patience = reduce_lr_patience
        self.reduce_lr_factor = reduce_lr_factor
        self.reduce_lr_min = reduce_lr_min
        self.lr_monitor = lr_monitor
        self.total_training_steps = total_training_steps

        self.save_hyperparameters(ignore=["model", "tokenizer"])

        if self.model is None:
            if self.model_init_config is None:
                raise ValueError("model_init_config is required to load model when model is None")
            self.model, self.tokenizer = load_model(**self.model_init_config)

    def training_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = outputs.loss.mean()
        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=True,
            logger=True,
            batch_size=batch["input_features"].size(0),
        )
        self.log(
            "train_loss_epoch",
            loss,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            batch_size=batch["input_features"].size(0),
        )
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = outputs.loss.mean()
        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=batch["input_features"].size(0),
            sync_dist=True,
        )
        return loss

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.learning_rate)

        if self.linear_lr_schedule and self.total_training_steps:
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=self.total_training_steps,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                    "monitor": "val_loss",
                },
            }

        if self.reduce_lr:
            scheduler = ReduceLROnPlateau(
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
                    "monitor": self.lr_monitor,
                },
            }

        return optimizer

    def on_save_checkpoint(self, checkpoint: dict[str, object]) -> None:
        checkpoint["das"] = dict(self.checkpoint_metadata)
        checkpoint["whisperseg"] = checkpoint_payload(
            self.model,
            self.tokenizer,
            int(getattr(self, "global_step", 0) or 0),
        )


class StopOnEventCallback(Callback):
    def __init__(self, stop_event, *, verbose: bool = False):
        self.stop_event = stop_event
        self.verbose = verbose
        self._logged = False

    def _request_stop(self, trainer) -> None:
        if self.stop_event is None or not self.stop_event.is_set():
            return
        if self.verbose and not self._logged:
            print("Cancellation requested. Stopping training after the current step.")
            self._logged = True
        trainer.should_stop = True

    def on_fit_start(self, trainer, pl_module):
        del pl_module
        self._request_stop(trainer)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        del pl_module, batch, batch_idx
        self._request_stop(trainer)

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        del pl_module, batch, batch_idx, dataloader_idx
        self._request_stop(trainer)


class EpochLogCallback(Callback):
    def _format_metric(self, value) -> str:
        if hasattr(value, "item"):
            value = value.item()
        return f"{float(value):.4f}"

    def on_train_epoch_start(self, trainer, pl_module):
        del pl_module
        current_epoch = int(trainer.current_epoch) + 1
        max_epochs = int(getattr(trainer, "max_epochs", 0) or 0)
        print(f"Epoch {current_epoch}/{max_epochs}" if max_epochs else f"Epoch {current_epoch}")

    def on_validation_epoch_end(self, trainer, pl_module):
        del pl_module
        if getattr(trainer, "sanity_checking", False):
            return
        metrics = getattr(trainer, "callback_metrics", {})
        parts = [
            f"{name}={self._format_metric(metrics[name])}"
            for name in ("train_loss", "val_loss", "train_loss_epoch")
            if name in metrics
        ]
        if parts:
            print(f"Epoch {int(trainer.current_epoch) + 1} complete: {', '.join(parts)}")


def train(
    model_folder: str,
    train_dataset_folder: str,
    *,
    initial_model_path: str,
    device: Literal["cpu", "cuda", "mps", "auto"] = "auto",
    n_device: int = 1,
    num_epochs: int = 10,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.0,
    max_length: int = 100,
    total_spec_columns: int = 1000,
    batch_size: int = 4,
    learning_rate: float = 3e-6,
    linear_lr_schedule: bool = True,
    reduce_lr: bool = False,
    reduce_lr_patience: int = 5,
    reduce_lr_factor: float = 0.1,
    reduce_lr_min: float = 1e-8,
    early_stopping: bool = True,
    early_stopping_patience: int = 3,
    seed: Optional[int] = None,
    weight_decay: float = 0.01,
    warmup_steps: int = 100,
    freeze_encoder: bool = False,
    encoder_dropout: float = 0.0,
    decoder_dropout: float = 0.0,
    num_workers: int = 0,
    ignore_class_names: bool = False,
    max_num_steps_per_epoch: Optional[int] = None,
    min_frequency: Optional[int] = None,
    frequency_scale: float = 1.0,
    spec_time_step: Optional[float] = None,
    include_labels: Optional[Sequence[str]] = None,
    audio_dataset: Optional[str] = None,
    data_samplerate_hz: Optional[float] = None,
    stop_event=None,
    emit_epoch_logs: bool = False,
    verbose: bool = False,
    checkpoint_filename: Optional[str] = None,
    checkpoint_metadata: Optional[Dict[str, Any]] = None,
):
    """Train the segmenter using PyTorch Lightning."""
    if linear_lr_schedule and reduce_lr:
        raise ValueError("WhisperSeg training supports either linear_lr_schedule or reduce_lr, not both.")

    if seed is not None:
        pl.seed_everything(seed, workers=True)

    os.makedirs(model_folder, exist_ok=True)

    model, tokenizer = load_model(
        initial_model_path=initial_model_path,
        total_spec_columns=total_spec_columns,
        encoder_dropout=encoder_dropout,
        decoder_dropout=decoder_dropout,
    )

    if freeze_encoder:
        for para in model.model.encoder.parameters():
            para.requires_grad = False

    train_dataset_folder = resolve_training_data_dir(train_dataset_folder)
    try:
        audio_path_list_train, label_path_list_train = get_audio_and_label_paths(
            train_dataset_folder,
            audio_dataset=audio_dataset,
            data_samplerate_hz=data_samplerate_hz,
        )
    except TypeError:
        # Some tests monkeypatch get_audio_and_label_paths with the historical one-argument signature.
        audio_path_list_train, label_path_list_train = get_audio_and_label_paths(train_dataset_folder)
    if not audio_path_list_train:
        raise ValueError(f"No paired readable audio and .json/.csv label files found in {train_dataset_folder}.")
    include_labels = _normalize_include_labels(include_labels)
    if include_labels:
        available_labels = set()
        for label_path in label_path_list_train:
            label = read_label(label_path, ignore_cluster=False)
            available_labels.update(str(cluster) for cluster in label["cluster"])
        _validate_include_labels(include_labels, available_labels)

    default_config = determine_default_config(
        audio_path_list_train,
        label_path_list_train,
        total_spec_columns,
        ignore_cluster=ignore_class_names,
        include_clusters=include_labels,
        min_frequency=min_frequency,
        frequency_scale=frequency_scale,
        spec_time_step=spec_time_step,
        audio_dataset=audio_dataset,
        data_samplerate_hz=data_samplerate_hz,
    )
    model.config.default_segmentation_config = default_config

    initial_codebook = {}
    cluster_codebook = get_cluster_codebook(
        label_path_list_train,
        initial_codebook,
        ignore_cluster=ignore_class_names,
        include_clusters=include_labels,
    )
    model.config.cluster_codebook = cluster_codebook

    train_records, val_records, _test_records = build_vocalseg_clip_records(
        audio_path_list_train,
        label_path_list_train,
        cluster_codebook=cluster_codebook,
        default_config=default_config,
        ignore_cluster=ignore_class_names,
        include_clusters=include_labels,
        total_spec_columns=total_spec_columns,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        audio_dataset=audio_dataset,
        data_samplerate_hz=data_samplerate_hz,
    )

    val_dataloader = None
    if validation_fraction > 0 and val_records:
        val_dataset = VocalSegDataset(
            val_records,
            tokenizer,
            max_length,
            total_spec_columns,
            model.config.species_codebook,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
            persistent_workers=num_workers > 0,
            pin_memory=False,
            timeout=1 if num_workers > 0 else 0,
        )

    training_dataset = VocalSegDataset(
        train_records,
        tokenizer,
        max_length,
        total_spec_columns,
        model.config.species_codebook,
    )

    train_dataloader = DataLoader(
        training_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
        persistent_workers=num_workers > 0,
        pin_memory=False,
        timeout=1 if num_workers > 0 else 0,
    )

    if len(train_dataloader) == 0:
        raise RuntimeError("Too few examples (less than a batch) for training.")

    steps_per_epoch = len(train_dataloader)
    if max_num_steps_per_epoch is not None:
        steps_per_epoch = min(int(max_num_steps_per_epoch), steps_per_epoch)
    num_epochs = int(num_epochs)
    max_num_iterations = steps_per_epoch * num_epochs
    monitor_metric = "val_loss" if val_dataloader is not None else "train_loss_epoch"

    lightning_module = WhisperSegLightningModule(
        model,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        linear_lr_schedule=linear_lr_schedule,
        warmup_steps=warmup_steps,
        reduce_lr=reduce_lr,
        reduce_lr_patience=reduce_lr_patience,
        reduce_lr_factor=reduce_lr_factor,
        reduce_lr_min=reduce_lr_min,
        lr_monitor=monitor_metric,
        total_training_steps=max_num_iterations,
        tokenizer=tokenizer,
        model_init_config={
            "initial_model_path": initial_model_path,
            "total_spec_columns": total_spec_columns,
            "encoder_dropout": encoder_dropout,
            "decoder_dropout": decoder_dropout,
        },
        checkpoint_metadata=checkpoint_metadata,
    )

    checkpoint_dir = os.path.join(model_folder, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=checkpoint_filename or "model",
        monitor=monitor_metric,
        mode="min",
        save_top_k=1,
        save_last=False,
        enable_version_counter=False,
    )
    callbacks = [
        checkpoint_callback,
        LearningRateMonitor(logging_interval="step"),
    ]
    if emit_epoch_logs:
        callbacks.extend((EpochLogCallback(), TrainingLogProgress()))
    elif verbose:
        callbacks.append(TQDMProgressBar())
    if stop_event is not None:
        callbacks.append(StopOnEventCallback(stop_event, verbose=emit_epoch_logs))

    if early_stopping:
        callbacks.append(
            EarlyStopping(
                monitor=monitor_metric,
                patience=early_stopping_patience,
                mode="min",
                verbose=verbose,
            )
        )

    trainer_kwargs = dict(
        accelerator="auto" if device == "auto" else ("gpu" if device == "cuda" else device),
        devices=n_device,
        max_epochs=num_epochs,
        logger=CSVLogger(save_dir=str(model_folder), name="logs"),
        callbacks=callbacks,
        enable_progress_bar=verbose and not emit_epoch_logs,
        enable_model_summary=verbose,
        log_every_n_steps=1,
        # suggest_integrations=False,
    )
    if max_num_steps_per_epoch is not None:
        trainer_kwargs["limit_train_batches"] = steps_per_epoch

    trainer = pl.Trainer(**trainer_kwargs)
    trainer.fit(
        lightning_module,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )
    return checkpoint_callback.best_model_path
