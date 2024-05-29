"""Utilities for logging training runs.

We currenlty have integrations for `tensorboard <https://www.tensorflow.org/tensorboard>`_, and `wandb.ai <https://wandb.ai>`_.
While tensorboard is integrated with tensorflow. To use wandb you'll have to install the wandb API:
"pip install wandb" or "conda install wandb -c conda-forge".
"""

import logging
import os
from typing import Optional, Dict

try:
    import wandb
    from wandb.keras import WandbCallback

    HAS_WANDB = True
except ImportError as e:
    logging.debug("Could not import neptune libraries.")
    HAS_WANDB = False
HAS_WANDB = True


class Wandb:
    """Utility class for logging to wandb.ai during training."""

    def __init__(
        self,
        project: Optional[str] = None,
        api_token: Optional[str] = None,
        entity: Optional[str] = None,
        params: Optional[Dict] = None,
        infer_from_env: bool = False,
    ):
        """
        Args:
            project (Optional[str], optional): Project to log to. Defaults to None.
            api_token (Optional[str], optional):  api token. Defaults to None.
            entity (Optional[str], optional):  Entity (user/team name). Defaults to None.
            params (Optional[Dict], optional): Dict to log to `config`. Defaults to None.
            infer_from_env (bool, optional): read project and api_token from environment variables
                                             WANDB_PROJECT and WANDB_API_TOKEN.
                                             Defaults to False.
        """
        if not HAS_WANDB:
            self.run = None
            logging.error("Could not import wandb in das.tracking.")
            return

        try:
            if project is None:
                project = os.environ["WANDB_PROJECT"]
            if api_token is None:
                api_token = os.environ["WANDB_API_TOKEN"]

            wandb.login(key=api_token)
            self.project = project
            self.entity = entity
            self.run = wandb.init(project=self.project, entity=self.entity, settings=wandb.Settings(start_method="fork"))

            if params is not None:
                wandb.config.update(params)

        except:
            self.run = None
            logging.exception("Wandb stuff went wrong.")

    def reinit(self, params=None):
        self.run = wandb.init(reinit=True, project=self.project, entity=self.entity)
        if params is not None:
            wandb.config.update(params)

    def finish(self):
        self.run.finish()

    def callback(self, save_model=False):  # -> Optional[WandbCallback]:
        """Get callback for auto-logging from tensorfow/keras."""
        # CHECK: Is callback re-usable across reinits?
        if self.run is not None:
            return WandbCallback(save_model=save_model)
        else:
            pass

    def log_test_results(self, report: Dict):
        """Log final classification result from test data.

        Args:
            report (Dict): dictionary containing the classification report.
        """
        if self.run is not None:
            wandb.summary.update(report)
