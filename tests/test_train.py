import das.train
import logging


logging.basicConfig(level=logging.INFO)

model, params = das.train.train(data_dir='tests/test.npy', save_dir='tests/test.res',
                          nb_epoch=4, fraction_data=0.1, version_data=True)

# load results from disk

# API_TOKEN = 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIyZjBkZGMwYS01OWQxLTQwNTctYWQ0OS04MmQzYzA2ODIxMjIifQ=='
# model, params = das.train.train(data_dir='tests/test.npy', save_dir='tests/test.res',
#                           nb_epoch=4, fraction_data=0.1,
#                           neptune_api_token=API_TOKEN, neptune_project='postpop/test')
print(params)
