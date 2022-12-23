# Experiment tracking with Weights & Biases

[Weights & Biases](https://wandb.ai) for tracking experiments and visualizing results. Great for comparing different model fits during manual parameter and automatic architecture tuning (see [example](/tutorials/architecture_tuning)).

## Setup Weights & Biases
- Create account at [https://wandb.ai](https://wandb.ai)
- Install the wandb python package and log in:
```shell
conda install wandb -c conda-forge
wandb login
```
- See [wandb docs](https://docs.wandb.ai/quickstart#1.-set-up-wandb) for details.

## Log your DAS runs:
- CLI:
```shell
das train OTHER_CLI_ARGS --wandb-token MY_SUPER_SECRET_TOKEN --wandb-entity django_reinhardt --wandb-project test_run
```
- Python:
```python
das.train.train(..., wandb-token="MY_SUPER_SECRET_TOKEN", wandb-entity="django_reinhardt", wandb_project="test_run")
```
- Args:
    - `wandb_api_token`: API token from wandb. See the [wandb docs](https://docs.wandb.ai/quickstart#1.-set-up-wandb) on how to generate the token.
    - `wandb_project`: Project to log to - useful for grouping runs.
    - `wandb_entity`: User or team name to log to.
