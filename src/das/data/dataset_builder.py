"""Build a DAS training dataset from annotated audio files."""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from .audio_dir import AudioDirDataModule, _annotation_bounds, _resample_audio_array, _split_lengths, load_audio_array


def make_training_dataset(
    data_folder: str | Path,
    store_folder: str | Path,
    *,
    split_by: str = "samples",
    validation_fraction: float = 0.2,
    test_fraction: float = 0.2,
    seed: int | None = None,
) -> Path:
    """Save annotated audio as a train/val/test NPY directory."""
    source = Path(data_folder).expanduser()
    destination = Path(store_folder).expanduser()
    if split_by not in {"samples", "files"}:
        raise ValueError("Split by must be 'samples' or 'files'.")
    if validation_fraction < 0 or test_fraction < 0 or validation_fraction + test_fraction >= 1:
        raise ValueError("Validation and test fractions must be non-negative and sum to less than 1.")
    if destination.exists():
        raise FileExistsError(f"Dataset folder already exists: {destination}")

    data = AudioDirDataModule(data_dir=str(source), batch_size=1, num_time_steps=512, val_ratio=0, test_ratio=0)
    if not data.annotated_audio_files:
        raise ValueError(f"No annotated audio files found in {source}.")

    infos = [data.source_audio_file_info_by_path[path.expanduser().resolve()] for path in data.annotated_audio_files]
    channel_counts = {int(info["channels"]) for info in infos}
    if len(channel_counts) != 1:
        raise ValueError("All audio files in a training dataset must have the same number of channels.")
    samplerate = max(int(info["samplerate"]) for info in infos)
    class_names = data.class_names
    class_index = {name: idx for idx, name in enumerate(class_names)}
    fractions = [1 - validation_fraction - test_fraction, validation_fraction, test_fraction]
    rng = np.random.default_rng(seed)
    parts = {name: {"x": [], "y": []} for name in ("train", "val", "test")}
    file_split_names = None
    if split_by == "files":
        counts = _split_lengths(len(infos), fractions)
        file_split_names = np.repeat(["train", "val", "test"], counts)[rng.permutation(len(infos))]

    for file_idx, (audio_file, annotations) in enumerate(zip(data.annotated_audio_files, data.annotations, strict=True)):
        audio, source_rate = load_audio_array(audio_file)
        if source_rate != samplerate:
            audio = _resample_audio_array(audio, source_rate, samplerate, axis=1 if audio.ndim == 2 else 0)
        audio = audio.T if audio.ndim == 2 else audio[:, None]
        labels = np.zeros((len(audio), len(class_names)), dtype=np.float32)
        for row in annotations.itertuples(index=False):
            start, stop = _annotation_bounds(row, 0.004)
            first = max(0, int(np.floor(start * samplerate)))
            last = min(len(audio), int(np.ceil(stop * samplerate)))
            if last > first:
                labels[first:last, class_index[str(row.name)]] = 1
        labels[:, 0] = np.clip(1 - labels[:, 1:].sum(axis=1), 0, 1)

        if file_split_names is None:
            counts = _split_lengths(len(audio), fractions)
            boundaries = np.cumsum([0, *counts])
            slices = {name: slice(int(boundaries[idx]), int(boundaries[idx + 1])) for idx, name in enumerate(parts)}
        else:
            slices = {str(file_split_names[file_idx]): slice(None)}
        for name, selection in slices.items():
            if len(audio[selection]):
                parts[name]["x"].append(audio[selection])
                parts[name]["y"].append(labels[selection])

    if not parts["train"]["x"]:
        raise ValueError("The training split is empty. Split by samples or add more annotated files.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="das-dataset-", dir=destination.parent) as temporary:
        output = Path(temporary) / "dataset"
        output.mkdir()
        np.save(
            output / "attrs.npy",
            {"samplerate_x_Hz": samplerate, "samplerate_y_Hz": samplerate,
             "class_names": class_names, "class_types": data.class_types},
            allow_pickle=True,
        )
        for name, arrays in parts.items():
            if arrays["x"]:
                split_folder = output / name
                split_folder.mkdir()
                np.save(split_folder / "x.npy", np.concatenate(arrays["x"]).astype(np.float32))
                np.save(split_folder / "y.npy", np.concatenate(arrays["y"]))
        output.rename(destination)
    return destination
