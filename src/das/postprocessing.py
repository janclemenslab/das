import das.io, das.predict, das.segment_utils
import numpy as np
import sklearn.metrics
from tqdm.autonotebook import tqdm
import warnings
from joblib import Parallel, delayed
import itertools
from typing import Optional, List, Tuple


# suppress UndefinedMetricWarnings in case of no preds
warnings.simplefilter("ignore")


def postprocess(probs, gap_dur=None, min_len=None):
    probs_post = das.predict.predict_segments(probs,
                                              segment_fillgap=gap_dur,
                                              segment_minlen=min_len)
    return probs_post['samples']


def score(labels_true, labels_pred):
    cr = sklearn.metrics.classification_report(labels_true,
                                               labels_pred,
                                               output_dict=True)
    return cr['macro avg']['f1-score']


def obj_fun(labels_true, probs_pred, gap_dur, min_len):
    warnings.simplefilter("ignore")
    labels_pred = postprocess(probs_pred, gap_dur, min_len)
    return score(labels_true, labels_pred)


def optimize_grid(labels_train_true,
                                 probs_train_pred,
                                 gap_durs,
                                 min_lens,
                                 n_jobs=-1):
    params = list(itertools.product(gap_durs, min_lens))
    f1_scores = Parallel(n_jobs=n_jobs)(
        delayed(obj_fun)(labels_train_true, probs_train_pred, gap_dur, min_len)
        for gap_dur, min_len in tqdm(params, total=len(params)))
    best_idx = np.argmax(f1_scores)
    best_gap_dur, best_min_len, best_score = params[best_idx][0], params[
        best_idx][1], f1_scores[best_idx]
    return best_gap_dur, best_min_len, best_score


def optimize(dataset_path: str, model_save_name: str, gap_durs: Optional[List[float]] = None,
             min_lens: Optional[List[float]] = None) -> Tuple[float, float, float, float]:
    """[summary]

    Args:
        dataset_path (str): [description]
        model_save_name (str): [description]
        gap_durs (Optional[List[float]], optional): [description]. Defaults to None.
        min_lens (Optional[List[float]], optional): [description]. Defaults to None.

    Returns:
        Tuple[float, float, float, float]: [description]
    """

    data = das.io.npy_dir.load(dataset_path)
    fs = data.attrs['samplerate_x_Hz']

    values = (2**(np.arange(-1, 10.5, 0.5))) / 1_000  # between 0.5 and 1024 ms

    # from seconds to samples
    if gap_durs is not None:
        gap_durs = (np.array(gap_durs) * fs).astype(np.int)
    else:
        gap_durs = (values * fs).astype(np.int)

    if min_lens is not None:
        min_lens = (np.array(min_lens) * fs).astype(np.int)
    else:
        min_lens = (values * fs).astype(np.int)

    _, _, probs_train_pred, _ = das.predict.predict(
        data['train']['x'], model_save_name=model_save_name)
    labels_train_true = postprocess(data['train']['y'])

    l = min(len(labels_train_true), len(probs_train_pred))

    # get the score before post-processing
    score_train_pre = score(labels_train_true[:l], postprocess(probs_train_pred)[:l])
    best_gap_dur, best_min_len, score_train = optimize_grid(
        labels_train_true[:l],
        probs_train_pred[:l],
        gap_durs=gap_durs,
        min_lens=min_lens)

    if len(data['val']['x']):
        _, _, probs_val_pred, _ = das.predict.predict(
            data['val']['x'], model_save_name=model_save_name)
        labels_val_true = postprocess(data['val']['y'])

        l = min(len(labels_val_true), len(probs_val_pred))
        score_val = obj_fun(labels_val_true[:l],
                            probs_val_pred[:l],
                            gap_dur=best_gap_dur,
                            min_len=best_min_len)
    else:
        score_val = None

    # from samples to seconds
    best_gap_dur, best_min_len = best_gap_dur / fs, best_min_len / fs

    return best_gap_dur, best_min_len, score_train_pre, score_train, score_val