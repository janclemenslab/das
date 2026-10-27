from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import sys


ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT = ROOT / "scripts" / "das_pyinstaller_entry.py"
APP_ENTRYPOINT = ROOT / "scripts" / "das_app_entry.py"
APP_ICON = ROOT / "assets" / "DAS.icns"
PRESETS = ROOT / "src" / "das" / "presets"

COPY_METADATA_PACKAGES = [
    "accelerate",
    "das",
    "huggingface-hub",
    "lightning",
    "magicgui",
    "nnaudio",
    "peft",
    "PySide6",
    "qtpy",
    "scikit-learn",
    "torch",
    "torchaudio",
    "torchvision",
    "transformers",
    "xarray-behave",
]

COLLECT_DATA_PACKAGES = [
    "magicgui",
    "transformers",
    "xarray_behave",
]

HIDDEN_IMPORTS = [
    "PySide6.QtOpenGL",
    "PySide6.QtOpenGLWidgets",
    "lightning.pytorch.callbacks",
    "lightning.pytorch.loggers",
    "magicgui.backends._qtpy",
    "magicgui.backends._qtpy.application",
    "magicgui.backends._qtpy.widgets",
    "nnAudio.features.mel",
    "nnAudio.features.stft",
    "pockets.iterators",
    "soundfile",
    "torchvision._C_stable",
    "torchvision.image_stable",
    "transformers.models.whisper",
]

EXCLUDED_MODULES = [
    "PyQt5",
    "PyQt6",
    "PySide2",
]


def build_pyinstaller_args(name: str, onefile: bool, app: bool = False) -> list[str]:
    args = [
        "--noconfirm",
        "--clean",
        "--name",
        name,
        "--distpath",
        str(ROOT / "dist"),
        "--workpath",
        str(ROOT / "build" / "pyinstaller"),
        "--specpath",
        str(ROOT / "build" / "pyinstaller"),
        "--paths",
        str(ROOT / "src"),
        "--add-data",
        f"{PRESETS}{os.pathsep}das/presets",
        "--windowed" if app and sys.platform == "darwin" else "--console",
        "--onefile" if onefile else "--onedir",
    ]
    if app:
        icon = APP_ICON if sys.platform == "darwin" else ROOT / "assets" / "das-icon.png"
        args.extend(["--icon", str(icon)])
        if sys.platform == "darwin":
            args.extend(["--osx-bundle-identifier", "org.janclemenslab.DAS"])
    for package in COPY_METADATA_PACKAGES:
        args.extend(["--copy-metadata", package])
    for package in COLLECT_DATA_PACKAGES:
        args.extend(["--collect-data", package])
    for module in HIDDEN_IMPORTS:
        args.extend(["--hidden-import", module])
    for module in EXCLUDED_MODULES:
        args.extend(["--exclude-module", module])
    args.append(str(APP_ENTRYPOINT if app else ENTRYPOINT))
    return args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a standalone DAS executable with PyInstaller.")
    parser.add_argument("--name", help="Executable name written under dist/ (default: DAS for --app, otherwise das).")
    parser.add_argument("--app", action="store_true", help="Build the double-clickable GUI (DAS.app on macOS, DAS.exe on Windows).")
    parser.add_argument(
        "--onefile",
        action="store_true",
        help="Build a single-file executable. The default one-dir build starts faster and is easier to debug.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the PyInstaller command without running it.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pyinstaller_args = build_pyinstaller_args(args.name or ("DAS" if args.app else "das"), args.onefile, args.app)
    command = ["pyinstaller", *pyinstaller_args]

    if args.dry_run:
        print(" ".join(shlex.quote(part) for part in command))
        return 0

    try:
        from PyInstaller.__main__ import run
    except ImportError:
        print("PyInstaller is not installed. Install it with: uv pip install pyinstaller", file=sys.stderr)
        return 1

    if sys.version_info >= (3, 14):
        print("Warning: build releases from Python 3.12 unless Python 3.14 support is intentional.", file=sys.stderr)

    run(pyinstaller_args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
