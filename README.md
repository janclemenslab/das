## Installation (MAC only for now)

```shell
conda create -n das.torch -y python=3.13 pip ffmpeg git uv
conda activate das.torch
uv pip install -e .
export KERAS_BACKEND="torch"; das
```
