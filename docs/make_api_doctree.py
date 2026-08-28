import os


ignore_modules = [
    "das.cli",
    "dsa._models",
]


API_HEAD = """\

.. _api:

Developer API
=============

.. toctree::
   :caption: API
   :maxdepth: 1

.. autosummary::
   :toctree: api

"""


def make_api_doctree():
    doctree = ""

    for root, dirs, files in os.walk("../src/das"):
        # remove leading "../src/"
        root = root[7:]

        for file in sorted(files):
            if file == "__init__.py" and root != "das":
                full = root.replace(os.sep, ".")
            elif file.endswith(".py") and not file.startswith("_"):
                full = os.path.join(root, file)
                full = full[:-3].replace(os.sep, ".")
            else:
                continue

            ignore = False
            for ignore_module in ignore_modules:
                if full.startswith(ignore_module):
                    ignore = True
                    break
            if not ignore:
                doctree += f"   {full}\n"

    # write file for api doc with header + doctree
    with open("api.rst", "w") as f:
        f.write("..\n  This file is auto-generated.\n\n")
        f.write(API_HEAD)
        f.write(doctree)


if __name__ == "__main__":
    make_api_doctree()
