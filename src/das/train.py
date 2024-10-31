"""Code for training networks."""

import time
import logging
import flammkuchen as fl
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow import keras
import os
import yaml
import dask.array as da
from typing import List, Optional, Tuple, Dict, Any, Union
from . import data, models, utils, predict, io, evaluate, tracking, data_hash, augmentation, postprocessing  # , timeseries

logger = logging.getLogger(__name__)

try:  # fixes cuDNN error when using LSTM layer
    import tensorflow as tf

    physical_devices = tf.config.list_physical_devices("GPU")
    if physical_devices:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, enable=True)
except Exception as e:
    logger.exception(e)


def train(
    *,
    data_dir: str,
    x_suffix: str = "",
    y_suffix: str = "",
    save_dir: str = "./",
    save_prefix: Optional[str] = None,
    save_name: Optional[str] = None,
    model_name: str = "tcn",
    nb_filters: int = 16,
    nb_kernels: Optional[int] = None,
    kernel_size: int = 16,
    nb_conv: int = 3,
    use_separable: List[bool] = False,
    nb_hist: int = 1024,
    ignore_boundaries: bool = True,
    batch_norm: bool = True,
    nb_pre_conv: int = 0,
    pre_nb_conv: Optional[int] = None,
    pre_nb_dft: int = 64,
    pre_kernel_size: int = 3,
    pre_nb_filters: int = 16,
    pre_nb_kernels: Optional[int] = None,
    upsample: bool = True,
    dilations: Optional[List[int]] = None,
    nb_lstm_units: int = 0,
    verbose: int = 2,
    batch_size: int = 32,
    nb_epoch: int = 400,
    learning_rate: Optional[float] = None,
    reduce_lr: bool = False,
    reduce_lr_patience: int = 5,
    fraction_data: Optional[float] = None,
    first_sample_train: Optional[int] = 0,
    last_sample_train: Optional[int] = None,
    first_sample_val: Optional[int] = 0,
    last_sample_val: Optional[int] = None,
    seed: Optional[int] = None,
    batch_level_subsampling: bool = False,
    augmentations: Optional[str] = None,
    tensorboard: bool = False,
    wandb_api_token: Optional[str] = None,
    wandb_project: Optional[str] = None,
    wandb_entity: Optional[str] = None,
    log_messages: bool = False,
    nb_stacks: int = 2,
    with_y_hist: bool = True,
    balance: bool = False,
    version_data: bool = True,
    post_opt: bool = False,
    post_opt_nb_workers: int = -1,
    post_opt_fill_gaps_min: float = 0.0005,
    post_opt_fill_gaps_max: float = 1.0,
    post_opt_fill_gaps_steps: int = 20,
    post_opt_min_len_min: float = 0.0005,
    post_opt_min_len_max: float = 1.0,
    post_opt_min_len_steps: int = 20,
    morph_kernel_duration: int = 32,
    morph_nb_kernels: int = 0,
    resnet_compute: bool = False,
    resnet_train: bool = False,
    tmse_weight: float = 0.0,
    _qt_progress: bool = False,
) -> Tuple[keras.Model, Dict[str, Any], keras.callbacks.History]:
    """Train a DAS network.

    Args:
        data_dir (str): Path to the directory or file with the dataset for training.
                        Accepts npy-dirs (recommended), h5 files or zarr files.
                        See documentation for how the dataset should be organized.
        x_suffix (str): Select dataset used for training in the data_dir by suffix (`y_` + X_SUFFIX).
                        Defaults to '' (will use the standard data 'x')
        y_suffix (str): Select dataset used as a training target in the data_dir by suffix (`y_` + Y_SUFFIX).
                        Song-type specific targets can be created with a training dataset,
                        Defaults to '' (will use the standard target 'y')
        save_dir (str): Directory to save training outputs.
                        The path of output files will constructed from the SAVE_DIR, an optional SAVE_PREFIX,
                        and the time stamp of the start of training.
                        Defaults to the current directory ('./').
        save_prefix (Optional[str]): Prepend to timestamp.
                           Name of files created will be start with SAVE_DIR/SAVE_PREFIX + "_" + TIMESTAMP
                           or with SAVE_DIR/TIMESTAMP if SAVE_PREFIX is empty.
                           Defaults to '' (empty).
        save_name (Optional[str]): Append to prefix.
                           Name of files created will be start with SAVE_DIR/SAVE_PREFIX + "_" + SAVE_NAME
                           or with SAVE_DIR/SAVE_NAME if SAVE_PREFIX is empty.
                           Defaults to the timestamp YYYYMMDD_hhmmss.
        model_name (str): Network architecture to use.
                          See das.models for a description of all models.
                          Defaults to `tcn`.
        nb_filters (int): Number of filters per layer.
                          Defaults to 16.
        kernel_size (int): Duration of the filters (=kernels) in samples.
                           Defaults to 16.
        nb_conv (int): Number of TCN blocks in the network.
                       Defaults to 3.
        use_separable (List[bool]): Specify which TCN blocks should use separable convolutions.
                                    Provide as a space-separated sequence of "False" or "True.
                                    For instance: "True False False" will set the first block in a
                                    three-block (as given by nb_conv) network to use separable convolutions.
                                    Defaults to False (no block uses separable convolutions).
        nb_hist (int): Number of samples processed at once by the network (a.k.a chunk duration).
                       Defaults to 1024 samples.
        ignore_boundaries (bool): Minimize edge effects by discarding predictions at the edges of chunks.
                                  Defaults to True.
        batch_norm (bool): Batch normalize.
                           Defaults to True.
        nb_pre_conv (int): Adds fronted with downsampling. The downsampling factor is `2**nb_pre_conv`.
                           The type of frontend depends on the model:
                           if model is `tcn`: adds a frontend of N conv blocks (conv-relu-batchnorm-maxpool2) to the TCN.
                           if model is `tcn_tcn`: adds a frontend of N TCN blocks to the TCN.
                           if model is `tcn_stft`: adds a trainable STFT frontend.
                           Defaults to 0 (no frontend, no downsampling).
        pre_nb_dft (int): Duration of filters (in samples) for the STFT frontend.
                          Number of filters is pre_nb_dft // 2 + 1.
                          Defaults to 64.
        pre_nb_filters (int): Number of filters per layer in the pre-processing TCN.
                              Defaults to 16. Deprecated.
        pre_kernel_size (int): Duration of filters (=kernels) in samples in the pre-processing TCN.
                               Defaults to 3. Deprecated.
        upsample (bool): whether or not to restore the model output to the input samplerate.
                         Should generally be True during training and evaluation but my speed up inference.
                         Defaults to True.
        dilations (List[int]): List of dilation rate, defaults to [1, 2, 4, 8, 16] (5 layer with 2x dilation per TCN block)
        nb_lstm_units (int): If >0, adds LSTM with `nb_lstm_units` LSTM units to the output of the stack of TCN blocks.
                             Defaults to 0 (no LSTM layer).
        verbose (int): Verbosity of training output (0 - no output during training, 1 - progress bar, 2 - one line per epoch).
                       Defaults to 2.
        batch_size (int): Batch size
                          Defaults to 32.
        nb_epoch (int): Maximal number of training epochs.
                        Training will stop early if validation loss did not decrease in the last 20 epochs.
                        Defaults to 400.
        learning_rate (Optional[float]): Learning rate of the model. Defaults should work in most cases.
                               Values typically range between 0.1 and 0.00001.
                               If None, uses model specific defaults: `tcn` 0.0001, `tcn_stft` and `tcn_tcn` 0.0005.
                               Defaults to None.
        reduce_lr (bool): Reduce learning rate when the validation loss plateaus.
                          Defaults to False.
        reduce_lr_patience (int): Number of epochs w/o a reduction in validation loss after which
                                  to trigger a reduction in learning rate.
                                  Defaults to 5 epochs.
        fraction_data (Optional[float]): Fraction of training and validation data to use.
                                         Defaults to 1.0.
                                         Overriden by setting all four *_sample_* args.
        first_sample_train (Optional[int]): Defaults to 0 (first sample in training dataset).
                                            Note 1: all four *_sample_* args must be set - otherwise they will be ignored.
                                            Note 2: Overrides fraction_data.
        last_sample_train (Optional[int]): Defaults to None (use last sample in training dataset).
        first_sample_val (Optional[int]): Defaults to 0 (first sample in validation dataset).
        last_sample_val (Optional[int]): Defaults to None (use last sample in validation dataset).
        seed (Optional[int]): Random seed to reproducibly select fractions of the data.
                              Defaults to None (no seed).
        batch_level_subsampling (bool): Select fraction of data for training from random subset of shuffled batches.
                                        If False, select a continuous chunk of the recording.
                                        Defaults to False.
        augmentations (Optional[str]): Path to yaml file or dictionary with the specification of augmentations.
                                       Defaults to None (no augmentations).
        tensorboard (bool): Write tensorboard logs to save_dir. Defaults to False.
        wandb_api_token (Optional[str]): API token for logging to wandb.
                                           Defaults to None (no logging to wandb).
        wandb_project (Optional[str]): Project to log to for wandb.
                                         Defaults to None (no logging to wandb).
        wandb_entity (Optional[str]): Entity to log to for wandb.
                                        Defaults to None (no logging to wandb).
        log_messages (bool): Sets terminal logging level to INFO.
                             Defaults to False (will follow existing settings).

        nb_stacks (int): Unused if model name is `tcn`, `tcn_tcn`, or `tcn_stft`. Defaults to 2.
        with_y_hist (bool): Unused if model name is `tcn`, `tcn_tcn`, or `tcn_stft`. Defaults to True.
        balance (bool): Balance data. Weights class-wise errors by the inverse of the class frequencies.
                        Defaults to False.
        version_data (bool): Save MD5 hash of the data_dir to log and params.yaml.
                             Defaults to True (set to False for large datasets since it can be slow).
        post_opt (bool): Optimize post processing (delete short detections, fill brief gaps).
                        Defaults to False.
        post_opt_nb_workers (int): Number of parallel processes during post_opt. Defaults to -1 (same number as cores).
        post_opt_fill_gaps_min (float): Defaults to 0.0005 seconds.
        post_opt_fill_gaps_max (float): Defaults to 1 second.
        post_opt_fill_gaps_steps (int): Defaults to 20.
        post_opt_min_len_min (float): Defaults to 0.0005 seconds.
        post_opt_min_len_max (float): Defaults to 1 second.
        post_opt_min_len_steps (int): Defaults to 20.

        morph_nb_kernels (int): Defaults to 0 (do not add morphological kernels).
        morph_kernel_duration (int): Defaults to 32.

        resnet_compute (bool): Defaults to False.
        resnet_train (bool): Defaults to False.

        tmse_weight (float): Defaults to 0.0.

    Returns:
        model (keras.Model)
        params (Dict[str, Any])
        history (keras.callbacks.History)
    """
    # _qt_progress: tuple of (multiprocessing.Queue, threading.Event)
    #        The queue is used to transmit progress updates to the GUI,
    #        the event is set in the GUI to stop training.
    if log_messages:
        logging.basicConfig(level=logging.INFO)

    if dilations is None:
        dilations = [1, 2, 4, 8, 16]

    # FIXME THIS IS NOT GREAT:
    sample_weight_mode = None
    data_padding = 0
    if with_y_hist:  # regression
        return_sequences = True
        stride = nb_hist
        y_offset = 0
        sample_weight_mode = "temporal"
        if ignore_boundaries:
            data_padding = int(
                np.ceil(kernel_size * nb_conv)
            )  # this does not completely avoid boundary effects but should minimize them sufficiently
            stride = stride - 2 * data_padding
    else:  # classification
        return_sequences = False
        stride = 1  # should take every sample, since sampling rates of both x and y are now the same
        y_offset = int(round(nb_hist / 2))
        upsample = False

    if stride <= 0:
        raise ValueError(
            "Stride <=0 - needs to be >0. Possible solutions: reduce kernel_size, increase nb_hist parameters, uncheck ignore_boundaries"
        )

    # update deprected:
    if pre_nb_conv is not None:
        nb_pre_conv = pre_nb_conv
    if nb_kernels is not None:
        nb_filters = nb_kernels

    if not upsample:
        output_stride = int(2**nb_pre_conv)
    else:
        output_stride = 1  # since we upsample output to original sampling rate. w/o upsampling: `output_stride = int(2**nb_pre_conv)` since each pre-conv layer does 2x max pooling

    if save_prefix is None:
        save_prefix = ""

    if len(save_prefix):
        save_prefix = save_prefix + "_"
    params = locals()
    del params["_qt_progress"]

    # remove learning rate param if not set so the value from the model def is used
    if params["learning_rate"] is None:
        del params["learning_rate"]

    if "_multi" in model_name:
        params["unpack_channels"] = True

    logger.info(f"Loading data from {data_dir}.")
    d = io.load(data_dir, x_suffix=x_suffix, y_suffix=y_suffix)

    params.update(d.attrs)  # add metadata from data.attrs to params for saving

    if version_data:
        logger.info("Versioning the data:")
        params["data_hash"] = data_hash.hash_data(data_dir)
        logger.info(f"   MD5 hash of {data_dir} is")
        logger.info(f"   {params['data_hash']}")

    sample_bounds_provided = (
        first_sample_train is not None
        and last_sample_train is not None
        and first_sample_val is not None
        and last_sample_val is not None
    )

    if fraction_data is not None and not sample_bounds_provided:
        if fraction_data > 1.0:  # seconds
            logger.info(
                f"{fraction_data} seconds corresponds to {fraction_data / (d['train']['x'].shape[0] / d.attrs['samplerate_x_Hz']):1.4f} of the training data."
            )
            fraction_data = np.min((fraction_data / (d["train"]["x"].shape[0] / d.attrs["samplerate_x_Hz"]), 1.0))
        elif fraction_data < 1.0:
            logger.info(f"Using {fraction_data:1.4f} of the training and validation data.")

    if (
        fraction_data is not None and not batch_level_subsampling and not sample_bounds_provided and fraction_data != 1.0
    ):  # train on a subset
        min_nb_samples = nb_hist * (batch_size + 2)  # ensure the generator contains at least one full batch
        first_sample_train, last_sample_train = data.sub_range(
            d["train"]["x"].shape[0], fraction_data, min_nb_samples, seed=seed
        )
        first_sample_val, last_sample_val = data.sub_range(d["val"]["x"].shape[0], fraction_data, min_nb_samples, seed=seed)
    elif sample_bounds_provided:
        logger.info("Using provided start/end samples:")
        logger.info(f"Train: {first_sample_train}:{last_sample_train}, Val: {first_sample_val}:{last_sample_val}.")

    # TODO clarify nb_channels, nb_freq semantics - always [nb_samples,..., nb_channels] -  nb_freq is ill-defined for 2D data
    params.update(
        {
            "nb_freq": d["train"]["x"].shape[1],
            "nb_channels": d["train"]["x"].shape[-1],
            "nb_classes": len(params["class_names"]),
            "first_sample_train": first_sample_train,
            "last_sample_train": last_sample_train,
            "first_sample_val": first_sample_val,
            "last_sample_val": last_sample_val,
        }
    )
    logger.info("Parameters:")
    logger.info(params)

    logger.info("Preparing data")
    if fraction_data is not None and batch_level_subsampling:  # train on a subset
        np.random.seed(seed)
        shuffle_subset = fraction_data
    else:
        shuffle_subset = None

    if augmentations:
        if isinstance(augmentations, str):
            logger.info(f"Initializing augmentations from {augmentations}.")
            aug_params = yaml.safe_load(open(augmentations, "r"))
        else:
            aug_params = augmentations
        params["augmentations"] = aug_params
        augs = augmentation.Augmentations.from_dict(params["augmentations"])
        logger.info(f"   Got {len(augs)} augmentations.")
    else:
        augs = None

    params["class_weights"] = None
    if balance:
        logger.info("Balancing classes:")
        logger.info("   Computing class weights.")
        params["class_weights"] = data.compute_class_weights(d["train"]["y"][first_sample_train:last_sample_train])
        logger.info(f"   {params['class_weights']}")

    data_gen = data.AudioSequence(
        d["train"]["x"],
        d["train"]["y"],
        shuffle=True,
        shuffle_subset=shuffle_subset,
        first_sample=first_sample_train,
        last_sample=last_sample_train,
        nb_repeats=1,
        batch_processor=augs,
        **params,
    )
    val_gen = data.AudioSequence(
        d["val"]["x"],
        d["val"]["y"],
        shuffle=False,
        shuffle_subset=shuffle_subset,
        first_sample=first_sample_val,
        last_sample=last_sample_val,
        **params,
    )
    # data_gen = timeseries.timeseries_dataset_from_array(d['train']['x'], d['train']['y'],
    #                               sequence_length=params['nb_hist'], sequence_stride=stride,
    #                               shuffle=True, batch_size=batch_size,
    #                               start_index=first_sample_train, end_index=last_sample_train)
    # val_gen = timeseries.timeseries_dataset_from_array(d['val']['x'], d['val']['y'],
    #                              sequence_length=params['nb_hist'], sequence_stride=stride,
    #                              shuffle=False, batch_size=batch_size,
    #                              start_index=first_sample_val, end_index=last_sample_val)

    logger.info(f"Training data:")
    logger.info(f"   {data_gen}")
    logger.info(f"Validation data:")
    logger.info(f"   {val_gen}")

    logger.info("Building network")
    try:
        model = models.model_dict[model_name](**params)
    except KeyError as e:
        logger.exception(e)
        raise ValueError(f"Model name was {model_name} but only {list(models.model_dict)} allowed.")

    logger.info(model.summary())
    os.makedirs(os.path.abspath(save_dir), exist_ok=True)
    if save_name is None:
        save_name = time.strftime("%Y%m%d_%H%M%S")
    save_name = "{0}/{1}{2}".format(save_dir, save_prefix, save_name)
    params["save_name"] = save_name
    logger.info(f"Will save to {save_name}.")

    # SET UP CALLBACKS
    checkpoint_save_name = save_name + "_model.h5"  # this will overwrite intermediates from previous epochs
    callbacks = [
        ModelCheckpoint(checkpoint_save_name, save_best_only=True, save_weights_only=False, monitor="val_loss", verbose=1),
        EarlyStopping(monitor="val_loss", patience=20, verbose=1),
    ]

    if reduce_lr:
        callbacks.append(ReduceLROnPlateau(patience=reduce_lr_patience, verbose=1))

    if _qt_progress:
        callbacks.append(utils.QtProgressCallback(nb_epoch, _qt_progress))

    if tensorboard:
        callbacks.append(TensorBoard(log_dir=save_dir))

    if wandb_api_token and wandb_project:  # could also get those from env vars!
        del params["wandb_api_token"]
        wandb = tracking.Wandb(wandb_project, wandb_api_token, wandb_entity, params)
        if wandb:
            callbacks.append(wandb.callback())
            params["wandb_run_name"] = None
            if hasattr(wandb.run, "name"):
                params["wandb_run_name"] = wandb.run.name

    utils.save_params(params, save_name)

    # TRAIN NETWORK
    logger.info("start training")
    fit_hist = model.fit(
        data_gen,
        epochs=nb_epoch,
        steps_per_epoch=min(len(data_gen), 1000),
        verbose=verbose,
        validation_data=val_gen,
        callbacks=callbacks,
    )

    tf.keras.backend.clear_session()

    # OPTIMIZE POSTPROCESSING
    if post_opt:
        logger.info("OPTIMIZING POSTPROCESSING:")

        gap_durs = np.geomspace(float(post_opt_fill_gaps_min), float(post_opt_fill_gaps_max), int(post_opt_fill_gaps_steps))
        min_lens = np.geomspace(float(post_opt_min_len_min), float(post_opt_min_len_max), int(post_opt_min_len_steps))

        best_gap_dur, best_min_len, scores = postprocessing.optimize(
            dataset_path=data_dir,
            model_save_name=save_name,
            gap_durs=gap_durs,
            min_lens=min_lens,
            nb_workers=post_opt_nb_workers,
        )

        logger.info(f"  Score on training data changed from {scores['train_pre']:1.4} to {scores['train']:1.4}.")
        if scores["val_pre"] is not None:
            logger.info(f"  Score on validation data changed from {scores['val_pre']:1.4} to {scores['val']:1.4}.")
        logger.info("  Optimal parameters for postprocessing:")
        logger.info(f"     gap_dur={best_gap_dur} seconds")
        logger.info(f"     min_len={best_min_len} seconds")

        params["post_opt"] = {
            "gap_dur": best_gap_dur,
            "min_len": best_min_len,
            "score_train": scores["train"],
            "score_val": scores["val"],
        }

        logger.info(f'   Updating params file "{save_name}_params.yaml" with the results.')
        utils.save_params(params, save_name)
        logger.info("DONE")

    # TEST
    # TODO use postprocessing params
    logger.info("TESTING:")
    if "test" not in d or len(d["test"]["x"]) < nb_hist:
        logger.info("   No test data - skipping final evaluation step.")
    else:
        logger.info(f"   Re-loading last best model from {checkpoint_save_name}.")
        model.load_weights(checkpoint_save_name)

        logger.info("   Predicting.")
        x_test, y_test, y_pred = evaluate.evaluate_probabilities(x=d["test"]["x"], y=d["test"]["y"], model=model, params=params)

        labels_test = predict.labels_from_probabilities(y_test)
        labels_pred = predict.labels_from_probabilities(y_pred)

        logger.info("   Evaluating.")
        conf_mat, report = evaluate.evaluate_segments(labels_test, labels_pred, params["class_names"], report_as_dict=True)
        logger.info(conf_mat)
        logger.info(report)
        params["conf_mat"] = conf_mat.tolist()
        params["report"] = report
        logger.info(f'   Updating params file "{save_name}_params.yaml" with the test results.')
        utils.save_params(params, save_name)

        if wandb_api_token and wandb_project:  # could also get those from env vars!
            wandb.log_test_results(report)

        save_filename = "{0}_results.h5".format(save_name)
        logger.info(f"   Saving to {save_filename}.")
        if "data_splits" in params:
            del params["data_splits"]  # paths with '/' break flammkuchen/pytables
        results_dict = {
            "fit_hist": dict(fit_hist.history),
            "confusion_matrix": conf_mat,
            "classification_report": report,
            "x_test": x_test,
            "y_test": y_test,
            # 'y_pred': np.array(y_pred),
            "labels_test": labels_test,
            # 'labels_pred': np.array(labels_pred),
            "params": params,
        }
        fl.save(save_filename, results_dict)
        da.to_hdf5(save_filename, {"/y_pred": y_pred, "/labels_pred": labels_pred})

    logger.info("DONE.")
    return model, params, fit_hist
