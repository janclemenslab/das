import defopt
import logging
import platform
from . import train, predict, evaluate

# from . import train_tune


logger = logging.getLogger(__name__)


def version():
    import das
    import sys
    import pandas as pd
    import numpy as np
    import scipy
    import h5py
    import sklearn as sk
    import tensorflow as tf

    try:
        import tensorflow.keras
    except (ModuleNotFoundError, ImportError):
        pass
    try:
        import keras
    except (ModuleNotFoundError, ImportError):
        pass

    import xarray as xr

    try:
        import xarray_behave as xb
        import pyqtgraph
        import qtpy
        import xarray_behave.gui.app

        has_gui = True
    except (ImportError, ModuleNotFoundError):
        has_gui = False

    gpu = len(tf.config.list_physical_devices("GPU")) > 0

    logger.info(f"  {platform.platform()}")
    logger.info(f"  DAS v{das.__version__}")
    logger.info(f"     GUI is {'' if has_gui else 'not'}available.")
    if has_gui:
        logger.info(f"     xarray-behave v{xb.__version__}")
        logger.info(f"     pyqtgraph v{pyqtgraph.__version__}")
        logger.info(f"     {qtpy.API_NAME} v{qtpy.PYQT_VERSION or qtpy.PYSIDE_VERSION}")
        logger.info(f"     Qt v{qtpy.QT_VERSION}")

    logger.info("")
    logger.info(f"  tensorflow v{tf.__version__}")
    if not hasattr(tensorflow.keras, "__version__"):
        logger.info(f"  keras v{keras.__version__}")
    else:
        logger.info(f"  keras v{tensorflow.keras.__version__}")
    logger.info(f"     GPU is {'' if gpu else 'not'} available.")
    logger.info("")
    logger.info(f"  python v{sys.version}")
    logger.info(f"  pandas v{pd.__version__}")
    logger.info(f"  numpy v{np.__version__}")
    logger.info(f"  h5py v{h5py.__version__}")
    logger.info(f"  scipy v{scipy.__version__}")
    logger.info(f"  scikit-learn v{sk.__version__}")
    logger.info(f"  xarray v{xr.__version__}")


def no_xb_gui():
    """Could not import the GUI. For instructions on how to install the GUI, check the docs janclemenslab.org/das/install.html."""
    logger.warning("Could not import the GUI.")
    logger.warning("For instructions on how to install the GUI,")
    logger.warning("check the docs janclemenslab.org/das/install.html.")


def main():
    """Command line interface for DAS."""
    subcommands = {
        "train": train.train,
        # "tune": train_tune.train,
        "predict": predict.cli_predict,
        "evaluate": evaluate.cli_evaluate,
        "version": version,
    }

    try:
        import xarray_behave.gui.app

        subcommands["gui"] = xarray_behave.gui.app.main_das
    except (ImportError, ModuleNotFoundError):
        logging.exception("No GUI avalaible.")
        # fall back to function that displays helpful instructions
        subcommands["gui"] = no_xb_gui

    logging.basicConfig(level=logging.INFO, force=True)
    defopt.run(subcommands, show_defaults=False)


if __name__ == "__main__":
    main()
