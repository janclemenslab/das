# Models and data

## Model files

- Conformer, TCN, and TweetyNet use DAS `.ckpt` files.
- Legacy DAS `*_model.h5` plus `*_params.yaml` models still load directly for prediction. `das convert-legacy` can convert them to `.ckpt`.
- WhisperSeg uses DAS `.ckpt` files for training and prediction. Select a named WhisperSeg AER model to download a compact FP16 checkpoint automatically, or pass a local checkpoint. Original WhisperSeg checkpoints and raw Hugging Face model IDs are not loaded at runtime.

## Training data

DAS trains directly from annotated WAV folders and continues to accept prebuilt `.npy`, H5, and Zarr DAS datasets. The existing dataset-generation path remains for other supported audio containers. Single- and multi-channel audio remain supported.

Embeddings, generic transfer learning, self-supervised training, automatic threshold grid search, and binary app downloads are outside this pre-release.
