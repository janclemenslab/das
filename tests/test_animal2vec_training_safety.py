from io import StringIO

import pytest

from scripts.pretrain_nightingale_animal2vec import check_gpu_step


def test_gpu_failures_stop_before_weight_update():
    check_gpu_step(74.0, StringIO('ordinary runtime warning\n'))
    for gradient, log in [(74.0, 'Error: command buffer exited with error status.\n'),
                          (1.5e7, ''), (float('nan'), '')]:
        with pytest.raises(RuntimeError, match='refusing to update model weights'):
            check_gpu_step(gradient, StringIO(log))
