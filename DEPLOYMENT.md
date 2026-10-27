# Deployment

Use PyInstaller to build platform-specific artifacts that include Python and the installed Python packages. Users of the built artifact do not need Python, conda, or uv installed.

Build on the same operating system and CPU architecture that you want to distribute. A macOS arm64 build should be made on Apple Silicon, a Windows build on Windows, and so on.

## Build

Create a clean release environment, then install DAS with the GUI dependencies and PyInstaller:

```shell
conda create -y -n das-release python=3.12 uv
conda activate das-release
uv pip install -e ".[gui]"
uv pip install pyinstaller
```

### macOS app

Install xarray-behave and build the double-clickable DAS GUI:

```shell
uv pip install xarray-behave
python scripts/build_executable.py --app
open dist/DAS.app
```

Double-clicking the app or running its executable without arguments opens the xarray-behave annotation GUI with its DAS train and predict integration. The same executable also accepts the regular DAS CLI commands:

```shell
dist/DAS.app/Contents/MacOS/DAS --help
dist/DAS.app/Contents/MacOS/DAS train --config train.yaml
dist/DAS.app/Contents/MacOS/DAS predict --config predict.yaml
dist/DAS.app/Contents/MacOS/DAS gui recording.wav
```

Zip the app without losing its bundle metadata:

```shell
ditto -c -k --sequesterRsrc --keepParent dist/DAS.app dist/DAS-macos-arm64.zip
```

PyInstaller ad-hoc signs local builds. Apps downloaded on another Mac need Developer ID signing and Apple notarization to open without a Gatekeeper workaround.

### Windows app

Build the double-clickable GUI on Windows:

```shell
python scripts/build_executable.py --app
dist\DAS\DAS.exe
```

Distribute the entire `dist/DAS` folder as a zip. The app includes Python and opens the annotation GUI directly.

### Command-line executable

Build the default one-dir executable:

```shell
python scripts/build_executable.py
```

Smoke-test the result:

```shell
dist/das/das --help
dist/das/das train --print-config
dist/das/das gui
```

For a single-file executable, use:

```shell
python scripts/build_executable.py --onefile
```

The one-file build is easier to distribute, but it starts more slowly because it extracts bundled libraries before each run. The one-dir build is usually better for debugging and for large PyTorch/Qt applications.

## Distribute

Zip the command-line artifact and share the zip:

```shell
ditto -c -k --sequesterRsrc --keepParent dist/das dist/das-macos-arm64.zip
```

For an installer, point the platform installer tool at `dist/das`. On macOS, signing and notarization are separate release steps; unsigned builds will be blocked or warned about on other machines.

## GitHub Releases

Run **Build distributable artifacts** from the repository's Actions tab to build and download the archives as workflow artifacts. Manual-run artifacts are kept for 30 days.

Publishing a GitHub release runs the same workflow. It builds and smoke-tests platform-specific bundles, then also uploads the archives to the release assets:

- `linux-x86_64` on `ubuntu-24.04`
- `windows-x86_64` on `windows-2022`
- `macos-arm64` on `macos-15` (`DAS.app`)

Linux and Windows release builds preinstall CPU-only PyTorch wheels before installing DAS. This keeps the release artifacts smaller and avoids bundling CUDA libraries into the default downloads.

## Caveats

- PyTorch, Qt, and Transformers make the artifact large.
- PyInstaller builds are not cross-compiled; build once per target platform.
- GPU support depends on the wheels installed in the release environment. The README install command currently resolves standard macOS wheels on macOS and CUDA wheels on Linux/Windows.
