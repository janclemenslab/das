from . import io, predict
import numpy as np
import sklearn.metrics
from tqdm.autonotebook import tqdm
import warnings
from joblib import Parallel, delayed
import itertools
from typing import List, Tuple, Dict, Union, Optional
import dask.array as da
import logging

logger = logging.getLogger(__name__)
# suppress UndefinedMetricWarnings in case of no preds
warnings.simplefilter("ignore")


def postprocess(class_probabilities, gap_dur=None, min_len=None, segment_dims=None, segment_names=None, segment_thres=None):
    probs_post = predict.predict_segments(
        class_probabilities,
        segment_fillgap=gap_dur,
        segment_minlen=min_len,
        segment_dims=segment_dims,
        segment_names=segment_names,
    )
    return probs_post["samples"]


def score(labels_true, labels_pred):
    return sklearn.metrics.f1_score(labels_true, labels_pred, average="macro")


def obj_fun(labels_true, probs_pred, gap_dur, min_len, segment_dims=None, segment_names=None):
    warnings.simplefilter("ignore")
    labels_pred = postprocess(probs_pred, gap_dur, min_len, segment_dims, segment_names)
    return score(labels_true, labels_pred)


def optimize_grid(
    labels_train_true, probs_train_pred, gap_durs, min_lens, segment_dims=None, segment_names=None, nb_workers=-1
):
    params = list(itertools.product(gap_durs, min_lens))
    f1_scores = Parallel(n_jobs=nb_workers)(
        delayed(obj_fun)(labels_train_true, probs_train_pred, gap_dur, min_len, segment_dims, segment_names)
        for gap_dur, min_len in tqdm(params, total=len(params))
    )
    # f1_scores = []
    # for gap_dur, min_len in tqdm(params, total=len(params)):
    #     val = obj_fun(labels_train_true, probs_train_pred, gap_dur, min_len, segment_dims, segment_names)
    #     f1_scores.append(val)
    best_idx = np.argmax(f1_scores)
    best_gap_dur, best_min_len, best_score = params[best_idx][0], params[best_idx][1], f1_scores[best_idx]
    return best_gap_dur, best_min_len, best_score, f1_scores


def optimize(
    dataset_path: str,
    model_save_name: str,
    gap_durs: List[float],
    min_lens: List[float],
    nb_workers: Optional[int] = -1,
) -> Tuple[float, float, Dict[str, Union[float, List[float]]]]:
    """[summary]

    Args:
        dataset_path (str): [description]
        model_save_name (str): [description]
        gap_durs (List[float], optional): in seconds.
        min_lens (List[float], optional): in seconds.
        nb_workers (int, optional): Number of parallel workers to use. Defaults to -1 (same as cores).

    Returns:
        Tuple[float, float, Dict[str, Union[float, List[float]]]]: [description]
    """

    data = io.npy_dir.load(dataset_path, memmap_dirs="all")
    fs = data.attrs["samplerate_x_Hz"]

    gap_durs = (gap_durs * fs).astype(int)
    min_lens = (min_lens * fs).astype(int)

    # get raw predictions to run post-processing on
    logger.info("   Generating raw predictions")
    _, segments, probs_train_pred, _ = predict.predict(data["train"]["x"], model_save_name=model_save_name, save_memory=True)
    labels_train_true = postprocess(
        da.from_array(data["train"]["y"]), segment_dims=segments["index"], segment_names=segments["names"]
    )

    ml = min(len(labels_train_true), len(probs_train_pred))

    # get the F1 score before post-processing
    labels_train_pred = postprocess(probs_train_pred, segment_dims=segments["index"], segment_names=segments["names"])
    score_train_pre = score(labels_train_true[:ml], labels_train_pred[:ml])

    # test all (gap_dur, min_len) combinations and choose the one maximizes the F1 score
    logger.info("   Grid search for optimal values")
    best_gap_dur, best_min_len, score_train, f1_scores = optimize_grid(
        labels_train_true[:ml],
        probs_train_pred[:ml],
        gap_durs=gap_durs,
        min_lens=min_lens,
        segment_dims=segments["index"],
        segment_names=segments["names"],
        nb_workers=nb_workers,
    )

    # validate found parameters on validation set
    if len(data["val"]["x"]):
        # get raw predictions to run post-processing on
        _, _, probs_val_pred, _ = predict.predict(data["val"]["x"], model_save_name=model_save_name, save_memory=True)
        labels_val_true = postprocess(
            da.from_array(data["val"]["y"]), segment_dims=segments["index"], segment_names=segments["names"]
        )

        ml = min(len(labels_val_true), len(probs_val_pred))

        # get the F1 score before post-processing
        labels_val_pred = postprocess(probs_val_pred, segment_dims=segments["index"], segment_names=segments["names"])
        score_val_pre = score(labels_val_true[:ml], labels_val_pred[:ml])

        # get score with best parameters
        score_val = obj_fun(
            labels_val_true[:ml],
            probs_val_pred[:ml],
            gap_dur=best_gap_dur,
            min_len=best_min_len,
            segment_dims=segments["index"],
            segment_names=segments["names"],
        )
    else:
        score_val_pre = None
        score_val = None

    # from samples to seconds
    best_gap_dur, best_min_len = float(best_gap_dur / fs), float(best_min_len / fs)

    scores = {
        "train_pre": float(score_train_pre),
        "train": float(score_train),
        "val_pre": float(score_val_pre),
        "val": float(score_val),
        "all_train": f1_scores,
    }

    return best_gap_dur, best_min_len, scores
