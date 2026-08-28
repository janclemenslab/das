# DAS PyTorch release handoff

## Verified state

- Branch: `das.torch`
- Prepared commit: `c24686dba90c6ccb324e916f576744acc2f5afa4`
- GitHub Actions: [all nine jobs passed](https://github.com/janclemenslab/das/actions/runs/33186039342) on Linux, macOS, and Windows with Python 3.12, 3.13, and 3.14.
- Local checks: 16 tests passed; all 63 notebooks parsed; clean wheel installs succeeded on Python 3.12 and 3.14 with Keras using the Torch backend.
- The working compatibility policy is one-way: old DAS code, arguments, imports, and TensorFlow-era H5 models must work with the Torch release. New artifacts do not need to work with the TensorFlow release.

## Release decisions

- `v0.32.13` is the final TensorFlow-backed release on `master`.
- Recommended first Torch-backed release: `0.33.0` with tag `v0.33.0`.
- Supported Python versions are 3.12 through 3.14; user documentation defaults to Python 3.14.
- Keep the strict-first model loading behavior: try strict layer-order loading, warn on failure, then retry with the loose compatibility flags.
- Morphology layers and automatic model tuning remain removed.

## Required work before publishing

1. Start from a fresh checkout and fetch the remote branches. Do not use the local `master` branch blindly: it currently contains the unpushed commit `65ec1c1`.
2. Open a pull request from `das.torch` into the current remote `master`. The branches diverged at `2969928`; remote `master` has 14 unique commits and `das.torch` has 12. Resolve the integration deliberately rather than replacing or force-pushing `master`.
3. Preserve the Torch versions of the package code, dependencies, installation documentation, and cross-platform test workflow. Preserve `master`'s deletion of `.github/workflows/publish.yml`: the copy on `das.torch` is an obsolete Conda workflow and must not survive the merge.
4. Add a short README migration note that `das==0.32.13` remains available for users who need TensorFlow.
5. Change `src/das/__init__.py` from `0.32.8` to `0.33.0`. Do not tag while it still reports `0.32.8`.
6. Add concise GitHub release notes covering the Torch backend, Python 3.12 minimum, legacy model/API compatibility, new installation command, and removal of morphology/tuning.

## Final release gates

- The merged/version-bumped commit passes the complete 3 OS × 3 Python GitHub Actions matrix.
- `uv build --wheel` succeeds from a clean checkout and the wheel metadata reports DAS 0.33.0 and `Requires-Python >=3.12`.
- Fresh environments can install that wheel on Python 3.12 and 3.14 using `uv pip install <wheel> --torch-backend=auto`; run `uv pip check`, `das version`, `das train --help`, and `das predict --help`.
- `pytest -q` passes from the installed release candidate, including `tests/test_legacy_api.py` and `tests/test_model_loading.py`.
- Re-run at least one real TensorFlow-era H5 model parity check, not only synthetic fixtures.
- Build the documentation and confirm that the README, installation page, and Colab notebook use Conda + uv, Python 3.14, and the Torch backend.

## Publish sequence

1. Merge the reviewed pull request into `master` and wait for the final matrix to pass on the merge commit.
2. Build and verify the wheel from that exact clean `master` commit.
3. Create and push the annotated tag:

   ```shell
   git tag -a v0.33.0 -m "Release DAS 0.33.0 (PyTorch backend)"
   git push origin v0.33.0
   ```

4. Publish the already-verified artifacts to PyPI with the project owner's credentials. There is currently no valid publishing workflow, so verify the destination and token before running `uv publish`.
5. Create the GitHub release from `v0.33.0` and attach or reference the same release notes.
6. In a new Python 3.14 Conda environment, install `das==0.33.0` from PyPI with `--torch-backend=auto` and repeat the CLI smoke checks.
7. Publish the updated documentation and verify the public installation and Colab pages.

PyPI versions cannot be overwritten. If upload fails after any artifact is accepted, diagnose first and release a new patch version rather than trying to replace `0.33.0`.
