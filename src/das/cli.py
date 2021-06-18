import defopt
import logging
from . import train, predict


def no_xb_gui():
    """Could not import the GUI. For instructions on how to install the GUI, check the docs janclemenslab.org/das/install.html."""
    print("Could not import the GUI.")
    print("For instructions on how to install the GUI,")
    print("check the docs janclemenslab.org/das/install.html.")


def main():
    """Command line interface for DeepSS."""
    subcommands = {'train': train.train, 'predict': predict.cli_predict}

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
