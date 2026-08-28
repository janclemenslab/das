## _DAS_ tutorials and documentation

The docs are published at [https://janclemenslab.org/das]().

Requires:

- `uv pip install . sphinx furo sphinx-inline-tabs myst-nb sphinx-panels sphinxcontrib-images`

### Build
Build the docs via `make clean html`. A fully-rendered HTML version will be built in `docs/_build/html/`.

### Publish
GitHub Actions publishes the docs when a GitHub release is published or the documentation workflow is run manually. Pull requests and pushes to `master` build the docs without deploying them.
