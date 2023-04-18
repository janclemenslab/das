## _DAS_ tutorials and documentation

The docs are published at [https://janclemenslab.org/das]().

Requires:

- `mamba install sphinx furo sphinx-inline-tabs ghp-import myst-nb sphinx-panels -c conda-forge`  # need the latest version for proper light/dark mode
- `pip install sphinxcontrib-images`

### Build
Build the docs via `make clean html`. A fully-rendered HTML version will be built in `docs/_build/html/`.

### Publish
Publish the book by running `make clean html push`. This will build the docs and push the built html files to [https://github.com/janclemenslab/das/tree/gh-pages]() and make it accessible via [https://janclemenslab.org/das]().
