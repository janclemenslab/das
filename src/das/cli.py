import defopt
import logging
from . import train, predict

def version():
    import das
    import sys
    import tensorflow.keras
    import pandas as pd
    import numpy as np
    import scipy
    import h5py
    import sklearn as sk
    import tensorflow as tf

    try:
        import xarray_behave.gui.app
        has_gui = True
    except (ImportError, ModuleNotFoundError):
        has_gui = False

    gpu = len(tf.config.list_physical_devices('GPU'))>0

    print(f"The following versions are available to DAS:")
    print(f"  DAS v{das.__version__}")
    print("     GUI is", "available" if has_gui else "NOT AVAILABLE")
    print("")
    print(f"  tensorflow v{tf.__version__}")
    print(f"  keras v{tensorflow.keras.__version__}")
    print("     GPU is", "available" if gpu else "NOT AVAILABLE")
    print("")
    print(f"  python v{sys.version}")
    print(f"  pandas v{pd.__version__}")
    print(f"  numpy v{np.__version__}")
    print(f"  h5py v{h5py.__version__}")
    print(f"  scipy v{scipy.__version__}")
    print(f"  scikit-learn v{sk.__version__}")


def no_xb_gui():
    """Could not import the GUI. For instructions on how to install the GUI, check the docs janclemenslab.org/das/install.html."""
    print("Could not import the GUI.")
    print("For instructions on how to install the GUI,")
    print("check the docs janclemenslab.org/das/install.html.")


def main():
    """Command line interface for DeepSS."""
    subcommands = {'train': train.train, 'predict': predict.cli_predict, 'version': version}

    try:
        import xarray_behave.gui.app
        subcommands['gui'] = xarray_behave.gui.app.main_das
    except (ImportError, ModuleNotFoundError):
        # fall back to function that displays helpful instructions
        subcommands['gui'] = no_xb_gui

    logging.basicConfig(level=logging.INFO)
    defopt.run(subcommands, show_defaults=False)


if __name__ == '__main__':
    main()
