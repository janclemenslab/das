# DAS PyTorch 0.33.0 pre-publication handoff

## Current state

- [`v0.32.13`](https://github.com/janclemenslab/das/releases/tag/v0.32.13) is published as the final TensorFlow-backed release. It removes morphology layers and model tuning.
- The PyTorch backend was merged by [PR #91](https://github.com/janclemenslab/das/pull/91) into `master` at `7de908e4a41cc2ea715275dfbf2608f15fe8200b`.
- The final post-merge [3 OS × 3 Python matrix](https://github.com/janclemenslab/das/actions/runs/33196234272) passed on that exact commit.
- DAS reports version `0.33.0` and requires Python 3.12 or newer.
- There is no `v0.33.0` tag, PyPI artifact, or GitHub release. PyPI still ends at `0.32.13`.

## Verification completed

- A clean archive of the merged commit built `das-0.33.0-py3-none-any.whl` with SHA-256 `af2957591dbe5584f569330eb1e26ad5841c346ebfcfdf6c8114d1a5cb93781e`.
- Fresh Python 3.12 and 3.14 environments installed that wheel with `uv` and the Torch backend.
- Both environments passed all 19 tests, including two real TensorFlow-era H5 models, plus `das version`, `das train --help`, `das predict --help`, and a GUI-window smoke test.
- Python 3.12 passes `uv pip check`.
- Python 3.14 runs successfully but fails `uv pip check`: `xarray-behave==0.37.4` pins `PySide6-Essentials==6.8.*`, whose metadata excludes Python 3.14.
- A clean documentation checkout builds HTML, but with 235 warnings. The public site is still a manual legacy Pages deployment from the `gh-pages` branch and was last updated in April 2025.

## Blockers before publication

### GUI dependency and Python 3.14

The current DAS metadata promises Python 3.14, but its GUI dependency graph does not. The published `xarray-behave` GUI also still exposes the removed morphology controls. Do not publish this mismatch.

Recommended fix: make a narrow `xarray-behave` compatibility release from its published `0.37.4` commit (`3433f0a99834`), not from current `xarray-behave/master`. Current master expects a `das.gui_app.DASConformerWindow` that DAS does not provide.

The compatibility release should:

1. Remove the obsolete morphology fields from the DAS training form.
2. Keep PySide 6.8 for Python below 3.14 and use a Python-3.14-compatible PySide release for Python 3.14.
3. Pass `uv pip check` and GUI train/predict smoke tests on Python 3.12 through 3.14.
4. Publish as `xarray-behave>=0.37.5`, then require that version from DAS.

The smaller alternative is to cap DAS at Python `<3.14` and change the documentation and CI to default to Python 3.13. The morphology form still needs a small `xarray-behave` patch either way.

### Documentation sources and deployment

The first cleanup pass reduces the clean-build warning count from 235 to 46 without changing documentation content:

- Delete the 49 tracked generated files under `docs/jupyter_execute/` (4.4 MB and about 15,000 duplicate lines).
- Stop tracking generated `docs/api/` stubs; `sphinx.ext.autosummary` recreates them during the build.
- Delete the unlinked, stale `continue_training.ipynb`, `inspect_dataset.ipynb`, and `tracking/fixing_identities.md` pages.
- Keep `docs/tutorials/colab.ipynb` as the one Colab source, update its badge, and delete the stale TensorFlow copy at `colab/colab.ipynb`.
- Track the 25 source GUI images currently hidden by `.gitignore`, fix the three incorrect bird-image names, and add `quickstart_bird.md` to the main toctree.
- Remove unused Sphinx configuration and invalid toctree entries, then fix the remaining broken links, duplicate figure labels, headings, and augmentation docstring warnings so the build passes with warnings treated as errors.

Replace the manual `ghp-import` push with one GitHub Pages workflow. It should build on pull requests and `master`, but deploy only on a published GitHub release or manual dispatch, using GitHub's official Pages artifact and deployment actions. Switch the repository Pages source from `gh-pages` to GitHub Actions and enable HTTPS. Keep the old branch as history; it does not need to be deleted.

## Ponytail cleanup included in the pre-publication change

- Remove the unused `torchvision` runtime dependency. DAS does not import it.
- Delete the completed TensorFlow-only `release-test.yml` workflow.
- Run the main test workflow on pull requests and pushes to `master`, not both for every feature-branch commit; PR #91 unnecessarily ran the same nine jobs twice.
- Make CI install the built wheel rather than an editable checkout, run `uv pip check`, and include the GUI smoke check.
- Ignore the now-obsolete root `env/` directory so deleted TensorFlow environment files do not reappear as untracked files.

Do not remove `models_legacy.py`, the legacy import shims, `utils_plot.py`, or the bundled Kapre/TCN code in 0.33.0. They are used by the promised old-model/API compatibility surface or by maintained notebooks. Do not mix a broad 494-finding Ruff modernization into this release.

## Execution plan

1. Release the narrow `xarray-behave` compatibility patch and verify its supported Python matrix.
2. Open one DAS pre-publication cleanup PR containing the dependency, CI, documentation-source, and Pages-workflow changes above.
3. Require a zero-warning documentation build and the 3 OS × 3 Python wheel-install matrix on that PR.
4. Rebuild the exact merged `master` wheel; on fresh Python 3.12 and 3.14 environments run `uv pip check`, all tests, CLI checks, GUI startup, and the real legacy-model parity checks.
5. Stop for an explicit go/no-go. Do not tag or upload yet.
6. After approval, create `v0.33.0`, publish the already-verified artifacts to PyPI, create the GitHub release, let that release deploy the Pages artifact, and verify clean PyPI installs plus the public installation and Colab pages.

PyPI versions cannot be overwritten. If an upload is partially accepted, diagnose first and use a new patch version instead of replacing `0.33.0`.
