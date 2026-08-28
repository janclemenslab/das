# DAS PyTorch 0.33.0 pre-publication handoff

## Current state

- [`v0.32.13`](https://github.com/janclemenslab/das/releases/tag/v0.32.13) is the published final TensorFlow-backed release. It removes morphology layers and model tuning.
- The PyTorch backend was merged by [PR #91](https://github.com/janclemenslab/das/pull/91) into `master` at `7de908e4a41cc2ea715275dfbf2608f15fe8200b` and its 3 OS × 3 Python matrix passed.
- The pre-publication cleanup is staged on `codex/torch-prepublish-cleanup` but is not pushed or merged yet.
- [`xarray-behave==0.37.5`](https://pypi.org/project/xarray-behave/0.37.5/) was published from [`8302c79`](https://github.com/janclemenslab/xarray-behave/commit/8302c79) and tagged `v0.37.5`. It is intentionally based on the published 0.37.4 commit, because current `xarray-behave/master` expects a `das.gui_app.DASConformerWindow` that DAS does not provide.
- `das==0.33.0` has not been tagged or published.

## Verified locally

- The `xarray-behave==0.37.5` wheel requires `PySide6-Essentials==6.10.*` and contains no morphology training fields.
- Fresh Python 3.12 and 3.14 environments install `xarray-behave==0.37.5` from PyPI with PySide 6.10.3, pass `uv pip check`, import the GUI, and pass the three tests used by its release workflow. The repository's full suite additionally has 9 assembly-test failures because its fixture data is not present; 13 other tests pass.
- Published SHA-256 hashes: wheel `450c24bdec77477de6073e043fb7bcd89c4e4afcdfbd1f7308afbfe714bf57ec`; sdist `ceca3037ba98304d738179c6943adcdf3e5d709a69f15f35b24ca3eb52819902`.
- A DAS 0.33.0 wheel built from the cleanup branch requires `xarray-behave>=0.37.5` and no longer requires `torchvision`.
- With the local `xarray-behave==0.37.5` wheel, fresh Python 3.12 and 3.14 environments both pass `uv pip check`, all 19 DAS tests, CLI checks, and a real GUI-window smoke test.
- A clean documentation archive builds successfully with 47 warnings, down from 235. Warning cleanup is deferred.

## Cleanup included

- Remove the unused `torchvision` dependency and obsolete TensorFlow release workflow.
- Avoid duplicate feature-branch CI runs, install the built wheel in CI, run `uv pip check`, and smoke-test GUI availability.
- Delete tracked generated Sphinx output and API stubs, three stale orphan pages, and the duplicate TensorFlow Colab notebook.
- Track the documentation's source images, repair the three missing bird-image references, and include the bird quickstart in the main navigation.
- Replace manual `gh-pages` pushes with a GitHub Pages workflow. Pull requests and `master` build without deploying; published releases and manual runs deploy.
- Ignore the obsolete root `env/` directory.

## Next steps

1. Push the DAS cleanup branch and open a PR. Require the 3 OS × 3 Python wheel-install matrix and documentation build to pass.
2. Merge the cleanup PR only when green. Then switch GitHub Pages from the legacy `gh-pages` source to GitHub Actions and enable HTTPS; keep the old branch as history.
3. Build the exact merged DAS wheel and repeat dependency, test, CLI, GUI, and legacy-model checks on fresh Python 3.12 and 3.14 environments.
4. Stop for an explicit DAS 0.33.0 go/no-go. Do not tag or upload before approval.
5. On approval, publish the already-verified DAS artifacts, create the GitHub release, allow that release to deploy the documentation, and verify PyPI and public documentation installs.

PyPI versions cannot be overwritten. If an upload is partially accepted, diagnose first and use a new patch version instead of replacing an existing version.

## Deferred to round two

- Fix the remaining 47 Sphinx warnings and then enforce warnings as errors.
- Apply the broad Ruff modernization separately; do not mix its 494 findings into the release.

Keep `models_legacy.py`, the legacy import shims, `utils_plot.py`, and bundled Kapre/TCN compatibility code. They support the promised old-model/API surface or maintained notebooks.
