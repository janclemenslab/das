import logging
import os
from typing import Optional, Dict
from sklearn.metrics import precision_recall_fscore_support

try:
    import nptune.new as neptune
    from neptune.new.integrations.tensorflow_keras import NeptuneCallback
    HAS_NEPTUNE = True
except ImportError as e:
    # logging.exception('Could not import neptune libraries.')
    HAS_NEPTUNE = False


class Poseidon():
    """Utility class for logging to neptune.ai in das.train.train."""

    def __init__(self, project: Optional[str] = None, api_token: Optional[str] = None,
                 params: Optional[Dict] = None, infer_from_env: bool = False):
        """Setup neptune run and log params

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
        """Log final classification result from test data."""
        if self.run is not None:
            self.run['classification_report'] = report
