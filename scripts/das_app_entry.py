from __future__ import annotations

import multiprocessing

from das.cli import cli_main


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(cli_main())
