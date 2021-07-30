import das.train

model, params = das.train.train(data_dir='test.npy', save_dir='test.res',
                          nb_epoch=4, fraction_data=0.1)

# load results from disk

# API_TOKEN = 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIyZjBkZGMwYS01OWQxLTQwNTctYWQ0OS04MmQzYzA2ODIxMjIifQ=='
# model, params = das.train.train(data_dir='test.npy', save_dir='test.res',
#                           nb_epoch=4, fraction_data=0.1,
#                           neptune_api_token=API_TOKEN, neptune_project='postpop/test')
print(params)
