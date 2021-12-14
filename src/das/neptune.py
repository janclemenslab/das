"""Utilities for logging to neptune.ai."""
import logging
import os
from typing import Optional, Dict


try:
    import neptune.new as neptune
    from neptune.new.integrations.tensorflow_keras import NeptuneCallback
    HAS_NEPTUNE = True
except ImportError as e:
    # logging.exception('Could not import neptune libraries.')
    HAS_NEPTUNE = False
HAS_NEPTUNE = True


try:
    import wandb as neptune
    from wandb.keras import WandbCallback
    HAS_WANDB = True
except ImportError as e:
    # logging.exception('Could not import neptune libraries.')
    HAS_WANDB = False
HAS_WANDB = True




class Poseidon():
    """Utility class for logging to neptune.ai in `das.train.train`."""

    def __init__(self, project: Optional[str] = None, api_token: Optional[str] = None,
                 params: Optional[Dict] = None, infer_from_env: bool = False):
        """Set up neptune run and log params.

        Args:
            project (Optional[str], optional): Project to log to. Defaults to None.
            api_token (Optional[str], optional): Neptune api token. Defaults to None.
            params (Optional[Dict], optional): Dict to log to `hyper-parameters`. Defaults to None.
            infer_from_env (bool, optional): read project and api_token from environment variables
                                             NEPTUNE_PROJECT and NEPTUNE_API_TOKEN.
                                             Defaults to False.
        """

        try:
            if project is None:
                project = os.environ['NEPTUNE_PROJECT']
            if api_token is None:
                api_token = os.environ['NEPTUNE_API_TOKEN']

            self.run = neptune.init(project, api_token)

            if params is not None:
                self.run['hyper-parameters'] = params

        except:
            self.run = None
            logging.exception('NEPTUNE stuff went wrong.')

    def callback(self):
        """Get callback for auto-logging from tensorfow/keras."""
        if self.run is not None:
            return NeptuneCallback(run=self.run, base_namespace='metrics')
        else:
            pass

    def log_test_results(self, report: Dict):
        """Log final classification result from test data.

        Args:
            report (Dict): dictionary containing the classification report.
        """
        if self.run is not None:
            self.run['classification_report'] = report


class Magic():
    """Utility class for logging to wandb in `das.train.train`."""

    def __init__(self, project: Optional[str] = None, api_token: Optional[str] = None,
                 entity: Optional[str] = None,
                 params: Optional[Dict] = None, infer_from_env: bool = False):
        """Set up wandb run and log params.

        Args:
            project (Optional[str], optional): Project to log to. Defaults to None.
            api_token (Optional[str], optional):  api token. Defaults to None.
            entity (Optional[str], optional):  Entity (user name). Defaults to None.
            params (Optional[Dict], optional): Dict to log to `hyper-parameters`. Defaults to None.
            infer_from_env (bool, optional): read project and api_token from environment variables
                                             WANDB_PROJECT and WANDB_API_TOKEN.
                                             Defaults to False.
        """

        try:
            if project is None:
                project = os.environ['WANDB_PROJECT']
            if api_token is None:
                api_token = os.environ['WANDB_API_TOKEN']

            wandb.login(key=api_token)
            self.run = wandb.init(project=project, entity=entity)

            if params is not None:
                wandb.config.update(params)

        except:
            self.run = None
            logging.exception('WANDB stuff went wrong.')

    def callback(self):
        """Get callback for auto-logging from tensorfow/keras."""
        if self.run is not None:
            return WandbCallback(run=self.run, base_namespace='metrics')
        else:
            pass

    def log_test_results(self, report: Dict):
        """Log final classification result from test data.

        Args:
            report (Dict): dictionary containing the classification report.
        """
        if self.run is not None:
            self.run.summary.update(report)