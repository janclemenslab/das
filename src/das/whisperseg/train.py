import math
import os
from typing import Any, Dict, Literal, Optional, Sequence

import lightning.pytorch as pl
import torch
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
from .model import _model_from_payload, _torch_load, checkpoint_payload, load_model


def _load_training_checkpoint(path):
    if not os.path.isfile(path):
        return None
    checkpoint = _torch_load(path)
    if not checkpoint.get("optimizer_states") or not ("whisperseg" in checkpoint or "whisperseg_training_config" in checkpoint):
        return None
    if "whisperseg_training" in checkpoint:
        payload = dict(checkpoint["whisperseg_training"])
        payload["model_state_dict"] = {
            name.removeprefix("model.base_model.model.").replace(".base_layer.", "."): value
            for name, value in checkpoint["state_dict"].items()
            if "lora_" not in name
        }
    elif "whisperseg" in checkpoint:
        payload = checkpoint["whisperseg"]
    else:
        return WhisperSegLightningModule.load_from_checkpoint(path, map_location="cpu", weights_only=True)
    model, tokenizer = _model_from_payload(payload)
    module = WhisperSegLightningModule(model, tokenizer=tokenizer, **checkpoint["hyper_parameters"])
    module.on_load_checkpoint(checkpoint)
    return module


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
        training_method: Literal["lora", "full"] = "full",
        freeze_encoder: bool = False,
        training_options: Optional[Dict[str, Any]] = None,
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
        self.training_method = training_method
        self.training_options = {} if training_options is None else dict(training_options)

        self.save_hyperparameters(ignore=["model", "tokenizer"])

        if self.model is None:
            if self.model_init_config is None:
                raise ValueError("model_init_config is required to load model when model is None")
            self.model, self.tokenizer = load_model(**self.model_init_config)

        if training_method == "lora":
            from peft import LoraConfig, get_peft_model

            self.model.half()
            self.model.config.use_cache = False
            self.model.model.decoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            self.model = get_peft_model(
                self.model,
                LoraConfig(
                    r=16,
                    lora_alpha=32,
                    lora_dropout=0.0,
                    bias="none",
                    target_modules=r"model\.decoder\.layers\.\d+\.(?:(?:self_attn|encoder_attn)\.(?:q_proj|k_proj|v_proj|out_proj)|fc1|fc2)",
                ),
            )
        elif freeze_encoder:
            for parameter in self.model.model.encoder.parameters():
                parameter.requires_grad = False

    def training_step(self, batch, batch_idx):
        batch["input_features"] = batch["input_features"].to(dtype=next(self.model.parameters()).dtype)
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
        batch["input_features"] = batch["input_features"].to(dtype=next(self.model.parameters()).dtype)
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
                "params": [
                    p for n, p in self.model.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.learning_rate,
            **({"eps": 1e-4} if self.training_method == "lora" else {}),
        )

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
        if self.training_method == "lora":
            # Keep resumable adapter state separate from the merged inference export.
            checkpoint["whisperseg_training_config"] = self.model.config.to_dict()
            payload = checkpoint_payload(self.model, self.tokenizer, int(self.global_step))
            del payload["model_state_dict"]
            checkpoint["whisperseg_training"] = payload
            return
        checkpoint["whisperseg"] = checkpoint_payload(
            self.model,
            self.tokenizer,
            int(getattr(self, "global_step", 0) or 0),
        )

    def on_load_checkpoint(self, checkpoint: dict[str, object]) -> None:
        config = (
            checkpoint["whisperseg_training_config"]
            if self.training_method == "lora"
            else checkpoint["whisperseg"]["model_config"]
        )
        self.model.config.update(config)


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
    training_method: Literal["lora", "full"] = "lora",
    device: Literal["cpu", "cuda", "mps", "auto"] = "auto",
    n_device: int = 1,
    num_epochs: int = 24,
    validation_fraction: float = 0.125,
    test_fraction: float = 0.0,
    max_length: int = 128,
    total_spec_columns: int = 1000,
    batch_size: int = 1,
    accumulate_grad_batches: int = 4,
    learning_rate: float = 1e-4,
    linear_lr_schedule: bool = False,
    reduce_lr: bool = True,
    reduce_lr_patience: int = 2,
    reduce_lr_factor: float = 0.5,
    reduce_lr_min: float = 1e-8,
    early_stopping: bool = True,
    early_stopping_patience: int = 5,
    seed: Optional[int] = 0,
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
    lightning_module = _load_training_checkpoint(initial_model_path)
    resuming = lightning_module is not None
    options = dict(
        batch_size=batch_size,
        accumulate_grad_batches=accumulate_grad_batches,
        max_length=max_length,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        seed=seed,
        ignore_class_names=ignore_class_names,
        include_labels=include_labels,
        audio_dataset=audio_dataset,
        data_samplerate_hz=data_samplerate_hz,
        early_stopping=early_stopping,
        early_stopping_patience=early_stopping_patience,
        max_num_steps_per_epoch=max_num_steps_per_epoch,
    )
    if resuming:
        options.update(lightning_module.training_options)
        training_method = lightning_module.training_method
    if training_method not in {"lora", "full"}:
        raise ValueError("training_method must be lora or full.")
    if options["accumulate_grad_batches"] < 1:
        raise ValueError("accumulate_grad_batches must be at least 1.")
    if not resuming and linear_lr_schedule and reduce_lr:
        raise ValueError("WhisperSeg training supports either linear_lr_schedule or reduce_lr, not both.")

    if options["seed"] is not None:
        pl.seed_everything(options["seed"], workers=True)

    os.makedirs(model_folder, exist_ok=True)

    if resuming:
        model, tokenizer = lightning_module.model, lightning_module.tokenizer
        total_spec_columns = model.config.total_spec_columns
    else:
        model, tokenizer = load_model(
            initial_model_path=initial_model_path,
            total_spec_columns=total_spec_columns,
            encoder_dropout=encoder_dropout,
            decoder_dropout=decoder_dropout,
        )

    train_dataset_folder = resolve_training_data_dir(train_dataset_folder)
    try:
        audio_path_list_train, label_path_list_train = get_audio_and_label_paths(
            train_dataset_folder,
            audio_dataset=options["audio_dataset"],
            data_samplerate_hz=options["data_samplerate_hz"],
        )
    except TypeError:
        # Some tests monkeypatch get_audio_and_label_paths with the historical one-argument signature.
        audio_path_list_train, label_path_list_train = get_audio_and_label_paths(train_dataset_folder)
    if not audio_path_list_train:
        raise ValueError(f"No paired readable audio and .json/.csv label files found in {train_dataset_folder}.")
    include_labels = _normalize_include_labels(options["include_labels"])
    if include_labels:
        available_labels = set()
        for label_path in label_path_list_train:
            label = read_label(label_path, ignore_cluster=False)
            available_labels.update(str(cluster) for cluster in label["cluster"])
        _validate_include_labels(include_labels, available_labels)

    default_config = (
        model.config.default_segmentation_config
        if resuming
        else determine_default_config(
            audio_path_list_train,
            label_path_list_train,
            total_spec_columns,
            ignore_cluster=options["ignore_class_names"],
            include_clusters=include_labels,
            min_frequency=min_frequency,
            frequency_scale=frequency_scale,
            spec_time_step=spec_time_step,
            audio_dataset=options["audio_dataset"],
            data_samplerate_hz=options["data_samplerate_hz"],
        )
    )
    model.config.default_segmentation_config = default_config

    initial_codebook = {}
    cluster_codebook = (
        model.config.cluster_codebook
        if resuming
        else get_cluster_codebook(
            label_path_list_train,
            initial_codebook,
            ignore_cluster=options["ignore_class_names"],
            include_clusters=include_labels,
        )
    )
    model.config.cluster_codebook = cluster_codebook

    train_records, val_records, _test_records = build_vocalseg_clip_records(
        audio_path_list_train,
        label_path_list_train,
        cluster_codebook=cluster_codebook,
        default_config=default_config,
        ignore_cluster=options["ignore_class_names"],
        include_clusters=include_labels,
        total_spec_columns=total_spec_columns,
        validation_fraction=options["validation_fraction"],
        test_fraction=options["test_fraction"],
        audio_dataset=options["audio_dataset"],
        data_samplerate_hz=options["data_samplerate_hz"],
    )

    val_dataloader = None
    if options["validation_fraction"] > 0 and val_records:
        val_dataset = VocalSegDataset(
            val_records,
            tokenizer,
            options["max_length"],
            total_spec_columns,
            model.config.species_codebook,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=options["batch_size"],
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
        options["max_length"],
        total_spec_columns,
        model.config.species_codebook,
    )

    train_dataloader = DataLoader(
        training_dataset,
        batch_size=options["batch_size"],
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
    if options["max_num_steps_per_epoch"] is not None:
        steps_per_epoch = min(int(options["max_num_steps_per_epoch"]), steps_per_epoch)
    num_epochs = int(num_epochs)
    max_num_iterations = math.ceil(steps_per_epoch / options["accumulate_grad_batches"]) * num_epochs
    monitor_metric = "val_loss" if val_dataloader is not None else "train_loss_epoch"

    lightning_module = (
        lightning_module
        if resuming
        else WhisperSegLightningModule(
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
            training_method=training_method,
            freeze_encoder=freeze_encoder,
            training_options=options,
        )
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

    if options["early_stopping"]:
        callbacks.append(
            EarlyStopping(
                monitor=monitor_metric,
                patience=options["early_stopping_patience"],
                mode="min",
                verbose=verbose,
            )
        )

    trainer_kwargs = dict(
        accelerator="auto" if device == "auto" else ("gpu" if device == "cuda" else device),
        devices=n_device,
        max_epochs=num_epochs,
        accumulate_grad_batches=options["accumulate_grad_batches"],
        gradient_clip_val=1.0 if training_method == "lora" else None,
        logger=CSVLogger(save_dir=str(model_folder), name="logs"),
        callbacks=callbacks,
        enable_progress_bar=verbose and not emit_epoch_logs,
        enable_model_summary=verbose,
        log_every_n_steps=1,
        # suggest_integrations=False,
    )
    if options["max_num_steps_per_epoch"] is not None:
        trainer_kwargs["limit_train_batches"] = steps_per_epoch

    trainer = pl.Trainer(**trainer_kwargs)
    trainer.fit(
        lightning_module,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        **({"ckpt_path": initial_model_path, "weights_only": True} if resuming else {}),
    )
    best_path = checkpoint_callback.best_model_path
    if training_method == "lora" and best_path:
        checkpoint = _torch_load(best_path)
        lightning_module.cpu()
        lightning_module.load_state_dict(checkpoint["state_dict"])
        merged = lightning_module.model.merge_and_unload(safe_merge=True).half().eval()
        merged.gradient_checkpointing_disable()
        merged.config.use_cache = True
        export_path = os.path.join(model_folder, os.path.basename(best_path))
        torch.save(
            {
                "das": dict(lightning_module.checkpoint_metadata),
                "whisperseg": checkpoint_payload(merged, tokenizer, checkpoint["global_step"]),
            },
            export_path,
        )
        return export_path
    return best_path
