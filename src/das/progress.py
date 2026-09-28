"""Training progress suitable for a plain-text GUI log."""

import math
import time

from lightning.pytorch.callbacks import Callback


class TrainingLogProgress(Callback):
    def on_train_epoch_start(self, trainer, pl_module) -> None:
        self._last_update = 0.0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx: int) -> None:
        total = trainer.num_training_batches
        if not math.isfinite(total) or total <= 0:
            return
        total = int(total)
        completed = batch_idx + 1
        now = time.monotonic()
        if completed not in (1, total) and now - self._last_update < 5:
            return
        self._last_update = now
        filled = min(20, int(20 * completed / total))
        bar = "#" * filled + "-" * (20 - filled)
        print(
            f"\rEpoch {trainer.current_epoch + 1}/{trainer.max_epochs}: "
            f"[{bar}] {completed}/{total} batches ({completed * 100 // total}%)",
            end="\n" if completed == total else "",
            flush=True,
        )
