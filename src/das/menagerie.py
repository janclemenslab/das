"""Utilities for interacting with DAS-Menagerie."""
from typing import Callable
import urllib.request
import tempfile
import numpy as np
# from github import Github
from typing import Optional


def _npz_loader(fname: str):
    out = {}
    with np.load(fname) as f:
        for key in f.keys():
            try:
                out[key] = f[key].astype(float)
            except ValueError:
                out[key] = f[key]
    return out


def load_data(url: str):
    with tempfile.NamedTemporaryFile() as fp:
        urllib.request.urlretrieve(url, fp.name)
        out = _npz_loader(fp.name)
    return out


# def download_model(name: str, to: str = '.'):
#     print(name, to)
#     url = name
#     urllib.request.urlretrieve(url, to)


# def list_models(gh_repo_url: str = "janclemenslab/das-menagerie", gh_access_token: Optional[str] = None):
#     G = Github(gh_access_token)
#     repo = G.get_repo(gh_repo_url)
#     releases = repo.get_releases()
#     trained_models = dict()
#     for release in releases:
#         name = release.tag_name
#         trained_models[name] = {}
#         trained_models[name]['zipball_url'] = release.zipball_url
#         trained_models[name]['models'] = []

#         # models: base_url, model_url, params_url
#         for asset in release.raw_data['assets']:
#             if asset['name'].endswith('_model.h5'):
#                 asset['basename'] = asset['name'][:-len('_model.h5')]
#                 # find accompanying params file
#                 for asset_param in release.raw_data['assets']:
#                     if asset_param['name'] == asset['basename'] + '_params.yaml':
#                         asset['params'] = asset_param
#                         asset['params_url'] = asset_param['url']
#                         asset['params_name'] = asset_param['name']
#                         asset['params_url_friendly'] = asset_param['browser_download_url']
#                 trained_models[name]['models'].append(asset)
#             elif asset['name'].endswith('_params.yaml'):
#                 asset_type = 'params'
#             elif asset['name'] == '/data.npz':
#                 asset_type = 'data'
#                 trained_models[name]['data'] = asset

#     return trained_models

# try:
#     trained_models = list_models()
# except Exception as e:
#     print(e)
