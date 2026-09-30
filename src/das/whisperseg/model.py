import os
import pickle
import tempfile
from pathlib import Path

import numpy as np
import torch
from transformers import GenerationConfig, WhisperConfig, WhisperTokenizer, WhisperForConditionalGeneration
import re
from sklearn.metrics import pairwise_distances
from sklearn.cluster import DBSCAN

from .audio_utils import WhisperSegFeatureExtractor, get_n_fft_given_sr
from scipy.stats import mode

from .utils import RATIO_DECODING_TIME_STEP_TO_SPEC_TIME_STEP

import threading


WHISPERSEG_CHECKPOINT_KEY = "whisperseg"
WHISPERSEG_CHECKPOINT_FORMAT = "das_whisperseg.checkpoint"
WHISPERSEG_CHECKPOINT_FORMAT_VERSION = 1

def _unwrap_model(model):
    try:
        return model.module
    except AttributeError:
        return model


def _serialize_tokenizer(tokenizer):
    with tempfile.TemporaryDirectory() as tmpdir:
        tokenizer.save_pretrained(tmpdir)
        tokenizer_files = {}
        for path in Path(tmpdir).rglob("*"):
            if path.is_file():
                tokenizer_files[path.relative_to(tmpdir).as_posix()] = path.read_bytes()
        return tokenizer_files


def _restore_tokenizer(tokenizer_files):
    with tempfile.TemporaryDirectory() as tmpdir:
        tokenizer_dir = Path(tmpdir)
        for relative_name, content in tokenizer_files.items():
            relative_path = Path(relative_name)
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise ValueError(f"Unsafe tokenizer file path in model bundle: {relative_name}")
            output_path = tokenizer_dir / relative_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(content)

        return WhisperTokenizer.from_pretrained(tmpdir, language="english")


def _torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=True, mmap=True)
    except pickle.UnpicklingError:
        from torch.serialization import safe_globals

        with safe_globals(_numpy_weights_only_safe_globals()):
            return torch.load(path, map_location=map_location, weights_only=True, mmap=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _numpy_weights_only_safe_globals():
    safe_globals = [np.dtype]
    try:
        safe_globals.append(np._core.multiarray.scalar)
    except AttributeError:
        safe_globals.append(np.core.multiarray.scalar)
    for dtype in ("bool", "int8", "int16", "int32", "int64", "float16", "float32", "float64", "complex64", "complex128"):
        safe_globals.append(type(np.dtype(dtype)))
    return safe_globals


def _make_weights_only_safe(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.dtype):
        return str(value)
    if isinstance(value, dict):
        return {key: _make_weights_only_safe(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_make_weights_only_safe(item) for item in value)
    if isinstance(value, list):
        return [_make_weights_only_safe(item) for item in value]
    return value


def checkpoint_payload(model, tokenizer, current_step):
    model = _unwrap_model(model)
    model.config.current_step = current_step
    return {
        "format": WHISPERSEG_CHECKPOINT_FORMAT,
        "format_version": WHISPERSEG_CHECKPOINT_FORMAT_VERSION,
        "model_class": model.__class__.__name__,
        "model_config": _make_weights_only_safe(model.config.to_dict()),
        "generation_config": (
            _make_weights_only_safe(model.generation_config.to_dict())
            if getattr(model, "generation_config", None) is not None
            else None
        ),
        "model_state_dict": model.state_dict(),
        "tokenizer_files": _serialize_tokenizer(tokenizer),
        "metadata": {"current_step": _make_weights_only_safe(current_step)},
    }


def _load_checkpoint_payload(model_path, map_location="cpu"):
    checkpoint = _torch_load(model_path, map_location=map_location)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"{model_path} is not a WhisperSeg checkpoint")
    payload = checkpoint.get(WHISPERSEG_CHECKPOINT_KEY)
    if not isinstance(payload, dict) or payload.get("format") != WHISPERSEG_CHECKPOINT_FORMAT:
        raise ValueError(f"{model_path} is not a WhisperSeg checkpoint")
    if payload.get("format_version") != WHISPERSEG_CHECKPOINT_FORMAT_VERSION:
        raise ValueError(f"Unsupported WhisperSeg checkpoint version: {payload.get('format_version')}")
    return payload


def is_model_checkpoint_path(model_path):
    path = Path(os.fspath(model_path)).expanduser()
    if not path.is_file() or path.suffix != ".ckpt":
        return False
    try:
        _load_checkpoint_payload(path)
    except Exception:
        return False
    return True


def load_model_checkpoint(model_path, map_location="cpu"):
    payload = _load_checkpoint_payload(model_path, map_location=map_location)
    config = WhisperConfig.from_dict(payload["model_config"])
    with torch.device("meta"):
        model = WhisperForConditionalGeneration(config)
    model.load_state_dict(payload["model_state_dict"], assign=True)
    model.tie_weights()
    if payload.get("generation_config") is not None:
        model.generation_config = GenerationConfig.from_dict(payload["generation_config"])
    tokenizer = _restore_tokenizer(payload["tokenizer_files"])
    return model, tokenizer


def load_inference_model(model_path):
    return load_model_checkpoint(Path(os.fspath(model_path)).expanduser())


def load_model(initial_model_path, total_spec_columns, encoder_dropout=0.0, decoder_dropout=0.0):
    model, tokenizer = load_model_checkpoint(Path(os.fspath(initial_model_path)).expanduser())
    model.config.max_source_positions = int(0.5 * total_spec_columns)
    with torch.no_grad():
        model.model.encoder.embed_positions.weight = torch.nn.Parameter(
            model.model.encoder.embed_positions.weight[: model.config.max_source_positions, :]
        )
        model.model.encoder.embed_positions.num_embeddings = model.config.max_source_positions

    model.config.total_spec_columns = total_spec_columns
    model.config.encoder_dropout = encoder_dropout
    model.config.decoder_dropout = decoder_dropout
    model.model.encoder.dropout = encoder_dropout
    for layer in model.model.encoder.layers:
        layer.dropout = encoder_dropout
    model.model.decoder.dropout = decoder_dropout
    for layer in model.model.decoder.layers:
        layer.dropout = decoder_dropout

    if not hasattr(model.config, "cluster_codebook"):
        model.config.cluster_codebook = {}

    if not hasattr(model.config, "species_codebook"):
        model.config.species_codebook = {
            "zebra_finch": "<|zebra_finch|>",
            "bengalese_finch": "<|bengalese_finch|>",
            "mouse": "<|mouse|>",
            "marmoset": "<|marmoset|>",
            "human": "<|human|>",
            ## set unknown for other species
            "unknown": "<|unknown|>",
            "animal": "<|animal|>",
        }

    tokenizer.add_tokens(["<|%d|>" % (i) for i in range(total_spec_columns + 1)], special_tokens=True)
    tokenizer.add_tokens([v for k, v in model.config.species_codebook.items()], special_tokens=True)

    return model, tokenizer


class SegmenterBase:
    def __init__(
        self,
    ):
        self.segment_matcher = re.compile(r"<\|([0-9]+)\|>(\d+?)<\|([0-9]+)\|>")
        self.total_spec_columns = None
        self.precision_bits = 3
        self.cluster_codebook = None
        self.default_segmentation_config = {}

    ### segmentation-related functions:
    def iter_sliced_audio_features(
        self, audio, sr, min_frequency, spec_time_step, num_trials, max_frequency=None
    ):
        feature_extractor = WhisperSegFeatureExtractor(
            sr,
            spec_time_step,
            min_frequency=min_frequency,
            max_frequency=max_frequency,
            chunk_length=max(30, int(np.ceil(spec_time_step * self.total_spec_columns))),
        )
        clip_duration = self.total_spec_columns * spec_time_step

        max_num_padding_samples = int(clip_duration * sr)
        audio_left_pad = np.zeros(max_num_padding_samples, dtype=np.float32)
        audio_clip_length = int(clip_duration * sr)

        for trial_id in range(num_trials):
            padding_time = np.round(clip_duration * trial_id / num_trials / spec_time_step) * spec_time_step
            num_padding_samples = int(padding_time * sr)
            audio_padded = np.concatenate([audio_left_pad[len(audio_left_pad) - num_padding_samples :], audio], axis=0)

            ## This loop must be executed once even for zero length audio
            for pos in range(0, max(len(audio_padded), 1), audio_clip_length):
                offset_time = pos / sr - padding_time

                audio_clip = audio_padded[pos : pos + audio_clip_length]
                audio_clip_padded = np.concatenate(
                    [audio_clip, np.zeros(audio_clip_length - len(audio_clip), dtype=np.float32)], axis=0
                )

                input_features = feature_extractor(audio_clip_padded, sampling_rate=sr, padding="do_not_pad")["input_features"][
                    0
                ]
                input_features = input_features[:, : self.total_spec_columns]

                if input_features.shape[1] > 0:
                    min_spec_value = input_features.min()
                else:
                    min_spec_value = 0
                # if input_features.shape[1] < self.total_spec_columns:
                input_features = np.concatenate(
                    [
                        input_features,
                        min_spec_value * np.ones((input_features.shape[0], self.total_spec_columns - input_features.shape[1])),
                    ],
                    axis=1,
                ).astype(np.float32)

                assert input_features.shape[1] == self.total_spec_columns

                yield trial_id, offset_time, input_features, len(audio_clip) / sr

    def get_sliced_audios_features(
        self, audio, sr, min_frequency, spec_time_step, num_trials, max_frequency=None
    ):
        return list(
            self.iter_sliced_audio_features(
                audio, sr, min_frequency, spec_time_step, num_trials, max_frequency=max_frequency
            )
        )

    ## This is used for WhisperSegmenter and WhisperSegmenterFast, not for WhisperSegmenterForEval
    def generate_segment_text(
        self,
        sliced_audios_features,
        batch_size,
        max_length,
        num_beams,
        top_k=1,
        top_p=1.0,
        length_penalty=1.0,
        status_monitor=None,
    ):
        generated_texts_dict = {}
        all_threads = []
        num_examples_per_thread = int(np.ceil(len(sliced_audios_features) / len(self.device_list)))
        for thread_id, pos in enumerate(range(0, len(sliced_audios_features), num_examples_per_thread)):
            t = threading.Thread(
                target=self.generate_segment_text_core,
                args=(
                    sliced_audios_features[pos : pos + num_examples_per_thread],
                    batch_size,
                    max_length,
                    num_beams,
                    top_k,
                    top_p,
                    length_penalty,
                    generated_texts_dict,
                    thread_id,
                    status_monitor if thread_id == 0 else None,  ## pass the status_monitor only to the first thread
                ),
            )
            t.start()
            all_threads.append(t)
        for t in all_threads:
            t.join()

        generated_text_list = []
        for thread_id in sorted(list(generated_texts_dict.keys())):
            generated_text_list += generated_texts_dict[thread_id]
        return generated_text_list

    def extract_segments(self, text, spec_time_step):
        inverse_cluster_codebook = {v: k for k, v in self.cluster_codebook.items()}
        segment_list = []
        match_res_list = self.segment_matcher.findall(text)
        for onset_text, cluster_id_text, offset_text in match_res_list:
            onset = int(onset_text) * spec_time_step * RATIO_DECODING_TIME_STEP_TO_SPEC_TIME_STEP
            offset = int(offset_text) * spec_time_step * RATIO_DECODING_TIME_STEP_TO_SPEC_TIME_STEP
            cluster_id = int(cluster_id_text)

            if cluster_id not in inverse_cluster_codebook:
                continue
            if offset - onset <= 0:
                continue

            cluster = inverse_cluster_codebook[cluster_id]
            segment_list.append([onset, offset, cluster])
        return segment_list

    def parse_generation(
        self,
        generated_text_list,
        sliced_audios_features,
        min_segment_length,
        audio_duration,
        spec_time_step,
        num_trials,
        eps,
        time_per_frame_for_voting,
        consolidation_method,
    ):
        ## convert generated text to on_offsets
        on_offset_list_of_trial = {}
        for count, generated_text in enumerate(generated_text_list):
            trial_id, offset_time, _, duration = sliced_audios_features[count]

            if trial_id not in on_offset_list_of_trial:
                on_offset_list_of_trial[trial_id] = []

            on_offsets_clip = self.extract_segments(generated_text, spec_time_step)
            for item in on_offsets_clip:
                item[0] += offset_time
                item[1] += offset_time

            on_offset_list_of_trial[trial_id].append(on_offsets_clip)

        ## merge (or concatenate) the on_offset of each trial separately
        merged_on_offset_list_of_trial = {}
        for trial_id in on_offset_list_of_trial:
            merged_on_offset_list_of_trial[trial_id] = []
            for on_offsets_clip in on_offset_list_of_trial[trial_id]:
                if (
                    len(merged_on_offset_list_of_trial[trial_id]) > 0
                    and len(on_offsets_clip) > 0
                    and merged_on_offset_list_of_trial[trial_id][-1][1] == on_offsets_clip[0][0]
                    and merged_on_offset_list_of_trial[trial_id][-1][2] == on_offsets_clip[0][2]
                ):
                    ## previous offset == current onset and the cluster type is the same, then we merge them

                    merged_on_offset_list_of_trial[trial_id][-1][1] = on_offsets_clip[0][1]
                    on_offsets_clip = on_offsets_clip[1:]
                merged_on_offset_list_of_trial[trial_id] += on_offsets_clip

        trials_results = []
        for trial_id in merged_on_offset_list_of_trial:
            for item in merged_on_offset_list_of_trial[trial_id]:
                item[0] = max(0, item[0])
                item[1] = min(item[1], audio_duration)

            merged_on_offset_list_of_trial[trial_id] = sorted(merged_on_offset_list_of_trial[trial_id], key=lambda x: x[0])
            merged_on_offset_list_of_trial[trial_id] = [
                item for item in merged_on_offset_list_of_trial[trial_id] if item[1] - item[0] >= min_segment_length
            ]

            pred_onsets, pred_offsets, pred_clusters = [], [], []
            for item in merged_on_offset_list_of_trial[trial_id]:
                pred_onsets.append(item[0])
                pred_offsets.append(item[1])
                pred_clusters.append(item[2])

            trials_results.append({"onset": list(pred_onsets), "offset": list(pred_offsets), "cluster": list(pred_clusters)})

        if num_trials == 1:
            final_prediction = trials_results[0]
        else:
            if consolidation_method == "clustering":
                min_samples = max(2, int(np.ceil(num_trials * 0.5)))
                final_prediction = self.consolidate_trials_by_clustering(trials_results, eps, min_samples)
            else:
                final_prediction = self.consolidate_trials_by_voting(trials_results, time_per_frame_for_voting)

        ##formating the final prediction
        final_prediction["onset"] = [float(np.round(t, self.precision_bits)) for t in final_prediction["onset"]]
        final_prediction["offset"] = [float(np.round(t, self.precision_bits)) for t in final_prediction["offset"]]

        return final_prediction

    ### multi-trial consolidation
    def custom_distance(self, segment1, segment2):
        onset_diff = abs(segment1[0] - segment2[0])
        offset_diff = abs(segment1[1] - segment2[1])
        return (onset_diff + offset_diff) / 2

    def consolidate_trials_by_clustering(self, trials, eps, min_samples):
        # Step 1: Create a list of all segments across all trials
        segments = []
        for trial_id, trial in enumerate(trials):
            for onset, offset, cluster in zip(trial["onset"], trial["offset"], trial["cluster"]):
                segments.append({"onset": onset, "offset": offset, "cluster": cluster, "trial": trial_id})
        if len(segments) == 0:
            return {"onset": [], "offset": [], "cluster": []}

        # Step 2: Compute pairwise distance matrix
        dist_matrix = pairwise_distances([[seg["onset"], seg["offset"]] for seg in segments], metric=self.custom_distance)
        segment_clusters = np.asarray([seg["cluster"] for seg in segments])
        dist_matrix[segment_clusters[:, None] != segment_clusters[None, :]] = eps + 1

        # Step 3: Cluster segments using DBSCAN
        db = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
        labels = db.fit_predict(dist_matrix)

        # Step 4: Merge segments within each cluster
        merged_segments = []
        for label in set(labels):
            if label != -1:  # Ignore noise
                group_segments = [seg for seg, lbl in zip(segments, labels) if lbl == label]
                if len(group_segments) == 0:
                    continue

                cluster_name_dict = {}
                for seg in group_segments:
                    cluster_name_dict[seg["cluster"]] = cluster_name_dict.get(seg["cluster"], 0) + 1
                ## get the most common cluster name
                cluster_name = sorted(list(cluster_name_dict.items()), key=lambda x: -x[1])[0][0]

                avg_onset = np.mean([seg["onset"] for seg in group_segments])
                avg_offset = np.mean([seg["offset"] for seg in group_segments])
                merged_segments.append({"onset": avg_onset, "offset": avg_offset, "cluster": cluster_name})

        merged_segments.sort(key=lambda x: x["onset"])

        final_pred = {
            "onset": [item["onset"] for item in merged_segments],
            "offset": [item["offset"] for item in merged_segments],
            "cluster": [item["cluster"] for item in merged_segments],
        }

        return final_pred

    def consolidate_trials_by_voting(self, trials, time_per_frame_for_voting):
        all_timestamps = []
        for trial in trials:
            all_timestamps += list(trial["onset"])
            all_timestamps += list(trial["offset"])
        if len(all_timestamps) == 0 or len(all_timestamps) % 2 != 0:
            return {"onset": [], "offset": [], "cluster": []}
        min_time = np.min(all_timestamps)
        max_time = np.max(all_timestamps)
        num_frames = int(np.round((max_time - min_time) / time_per_frame_for_voting))

        all_frame_wise_predictions = []
        for trial in trials:
            frame_wise_prediction = np.ones(num_frames) * -1
            for pos in range(len(trial["onset"])):
                onset = trial["onset"][pos] - min_time
                offset = trial["offset"][pos] - min_time
                cluster_id = self.cluster_codebook[trial["cluster"][pos]]
                frame_wise_prediction[
                    int(np.round(onset / time_per_frame_for_voting)) : int(np.round(offset / time_per_frame_for_voting))
                ] = cluster_id
            all_frame_wise_predictions.append(frame_wise_prediction)

        all_frame_wise_predictions = np.asarray(all_frame_wise_predictions)
        voted_frame_wise_prediction, counts = mode(all_frame_wise_predictions, axis=0)
        voted_frame_wise_prediction_right_pad = np.array(voted_frame_wise_prediction.tolist() + [-1])
        voted_frame_wise_prediction_left_pad = np.array([-1] + voted_frame_wise_prediction.tolist())
        event_positions = np.argwhere(voted_frame_wise_prediction_right_pad - voted_frame_wise_prediction_left_pad != 0)[:, 0]

        final_onsets = []
        final_offsets = []
        final_clusters = []

        inverse_cluster_codebook = {v: k for k, v in self.cluster_codebook.items()}

        for idx in range(0, len(event_positions) - 1):
            onset_pos = event_positions[idx]
            offset_pos = event_positions[idx + 1]
            cluster_id = int(np.round(np.mean(voted_frame_wise_prediction[onset_pos:offset_pos])))

            if cluster_id == -1:
                continue

            final_onsets.append(onset_pos * time_per_frame_for_voting + min_time)
            final_offsets.append(offset_pos * time_per_frame_for_voting + min_time)
            final_clusters.append(inverse_cluster_codebook[cluster_id])

        final_pred = {"onset": final_onsets, "offset": final_offsets, "cluster": final_clusters}

        return final_pred

    def generate_segment_records(
        self,
        audio,
        sr,
        min_frequency,
        spec_time_step,
        num_trials,
        batch_size,
        max_length,
        num_beams,
        top_k,
        top_p,
        length_penalty,
        status_monitor,
        max_frequency=None,
    ):
        sliced_audios_features = self.get_sliced_audios_features(
            audio, sr, min_frequency, spec_time_step, num_trials, max_frequency=max_frequency
        )
        generated_text_list = self.generate_segment_text(
            sliced_audios_features,
            batch_size,
            max_length,
            num_beams,
            top_k,
            top_p,
            length_penalty,
            status_monitor,
        )
        return generated_text_list, sliced_audios_features

    @torch.no_grad()
    def segment(
        self,
        audio,
        sr,
        min_frequency=None,
        max_frequency=None,
        spec_time_step=None,
        min_segment_length=None,
        eps=None,  ## for DBSCAN clustering
        time_per_frame_for_voting=None,  ## for voting
        consolidation_method="clustering",
        max_length=128,
        batch_size=4,
        num_trials=1,
        num_beams=4,
        top_k=1,
        top_p=1.0,
        length_penalty=1.0,
        status_monitor=None,
    ):
        defaults = self.default_segmentation_config
        rate_defaults = defaults.get("sample_rate_configs", {}).get(str(sr), {})
        if min_frequency is None:
            min_frequency = rate_defaults.get("min_frequency", defaults.get("min_frequency", 0))
        if spec_time_step is None:
            spec_time_step = rate_defaults.get("spec_time_step", defaults.get("spec_time_step", 0.0025))

        if min_segment_length is None:
            min_segment_length = defaults.get("min_segment_length", spec_time_step * RATIO_DECODING_TIME_STEP_TO_SPEC_TIME_STEP)
        if eps is None:
            eps = defaults.get("eps", spec_time_step * RATIO_DECODING_TIME_STEP_TO_SPEC_TIME_STEP * 4)
        if time_per_frame_for_voting is None:
            time_per_frame_for_voting = spec_time_step

        generated_text_list, sliced_audios_features = self.generate_segment_records(
            audio,
            sr,
            min_frequency,
            spec_time_step,
            num_trials,
            batch_size,
            max_length,
            num_beams,
            top_k,
            top_p,
            length_penalty,
            status_monitor,
            max_frequency=max_frequency,
        )
        final_prediction = self.parse_generation(
            generated_text_list,
            sliced_audios_features,
            min_segment_length,
            len(audio) / sr,
            spec_time_step,
            num_trials,
            eps,
            time_per_frame_for_voting,
            consolidation_method,
        )

        ## eliminate the FFT blurring effect
        n_fft = get_n_fft_given_sr(sr)
        time_delta = n_fft / 2 / sr

        corrected_onsets = []
        corrected_offsets = []
        for onset, offset in zip(final_prediction["onset"], final_prediction["offset"]):
            corr_onset = onset + time_delta
            corr_offset = offset - time_delta
            if corr_onset > corr_offset:
                corr_onset = (onset + offset) / 2
                corr_offset = (onset + offset) / 2
            corrected_onsets.append(corr_onset)
            corrected_offsets.append(corr_offset)

        final_prediction["onset"] = corrected_onsets
        final_prediction["offset"] = corrected_offsets

        ## remove the duplicated segments (onset offset are the same and cluster name is also the same)
        if len(final_prediction["onset"]) > 0:
            clean_onset_list, clean_offset_list, clean_cluster_list = [], [], []
            for onset, offset, cluster in sorted(
                zip(final_prediction["onset"], final_prediction["offset"], final_prediction["cluster"]), key=lambda x: x[0]
            ):
                if (
                    len(clean_onset_list) == 0
                    or onset != clean_onset_list[-1]
                    or offset != clean_offset_list[-1]
                    or cluster != clean_cluster_list[-1]
                ):
                    clean_onset_list.append(onset)
                    clean_offset_list.append(offset)
                    clean_cluster_list.append(cluster)

            final_prediction["onset"] = clean_onset_list
            final_prediction["offset"] = clean_offset_list
            final_prediction["cluster"] = clean_cluster_list

        return final_prediction

    ### evaluation-related functions
    def compute_syllable_score(self, prediction_on_offset_list, label_on_offset_list, tolerance):
        n_positive_in_prediction = len(prediction_on_offset_list)
        n_positive_in_label = len(label_on_offset_list)

        n_true_positive = 0
        for pred_onset, pred_offset, pred_cluster in prediction_on_offset_list:
            is_matched = False
            for count, (label_onset, label_offset, label_cluster) in enumerate(label_on_offset_list):
                if (
                    np.abs(pred_onset - label_onset) <= tolerance
                    and np.abs(pred_offset - label_offset) <= tolerance
                    and pred_cluster == label_cluster
                ):
                    # print( (pred_onset, pred_offset), (label_onset, label_offset) )
                    n_true_positive += 1
                    is_matched = True
                    break  # early stop for the predicted value
            if is_matched:
                ## remove the already matched syllable from the ground-truth
                label_on_offset_list.pop(count)

        return n_true_positive, n_positive_in_prediction, n_positive_in_label

    def segment_score(self, prediction, label, target_cluster=None, tolerance=None):
        if tolerance is None:
            tolerance = self.default_segmentation_config.get("spec_time_step", 0.0025) * 4

        prediction_on_offset_list = []
        for pos in range(len(prediction["onset"])):
            if target_cluster is None or str(target_cluster) == str(prediction["cluster"][pos]):
                prediction_on_offset_list.append(
                    [prediction["onset"][pos], prediction["offset"][pos], str(prediction["cluster"][pos])]
                )

        label_on_offset_list = []
        for pos in range(len(label["onset"])):
            if target_cluster is None or str(target_cluster) == str(label["cluster"][pos]):
                label_on_offset_list.append([label["onset"][pos], label["offset"][pos], str(label["cluster"][pos])])

        if target_cluster is not None and len(label_on_offset_list) == 0:
            print(
                "Warning: the specified target cluster '%s' does not exist in the ground-truth labels." % (str(target_cluster))
            )

        TP, P_pred, P_label = self.compute_syllable_score(prediction_on_offset_list, label_on_offset_list, tolerance)

        precision = TP / max(P_pred, 1e-12)
        recall = TP / max(P_label, 1e-12)
        f1 = 2 / (1 / max(precision, 1e-12) + 1 / max(recall, 1e-12))

        return TP, P_pred, P_label, precision, recall, f1

    def frame_score(self, prediction, label, target_cluster=None, time_per_frame_for_scoring=None):
        if time_per_frame_for_scoring is None:
            time_per_frame_for_scoring = min(0.001, self.default_segmentation_config.get("spec_time_step", 0.0025))

        prediction_segments = prediction
        label_segments = label

        prediction_segments["cluster"] = list(map(str, prediction_segments["cluster"]))
        label_segments["cluster"] = list(map(str, label_segments["cluster"]))

        cluster_to_id_mapper = {}
        for cluster in list(prediction_segments["cluster"]) + list(label_segments["cluster"]):
            if cluster not in cluster_to_id_mapper:
                cluster_to_id_mapper[cluster] = len(cluster_to_id_mapper)

        all_timestamps = (
            list(prediction_segments["onset"])
            + list(prediction_segments["offset"])
            + list(label_segments["onset"])
            + list(label_segments["offset"])
        )
        if len(all_timestamps) == 0:
            max_time = 1.0
        else:
            max_time = np.max(all_timestamps)

        num_frames = int(np.round(max_time / time_per_frame_for_scoring)) + 1

        frame_wise_prediction = np.ones(num_frames) * -1
        for idx in range(len(prediction_segments["onset"])):
            onset_pos = int(np.round(prediction_segments["onset"][idx] / time_per_frame_for_scoring))
            offset_pos = int(np.round(prediction_segments["offset"][idx] / time_per_frame_for_scoring))
            frame_wise_prediction[onset_pos:offset_pos] = cluster_to_id_mapper[prediction_segments["cluster"][idx]]

        frame_wise_label = np.ones(num_frames) * -1
        for idx in range(len(label_segments["onset"])):
            onset_pos = int(np.round(label_segments["onset"][idx] / time_per_frame_for_scoring))
            offset_pos = int(np.round(label_segments["offset"][idx] / time_per_frame_for_scoring))
            frame_wise_label[onset_pos:offset_pos] = cluster_to_id_mapper[label_segments["cluster"][idx]]

        if target_cluster is None:
            TP = np.logical_and(frame_wise_label != -1, frame_wise_prediction == frame_wise_label).sum()
            P_in_pred = (frame_wise_prediction != -1).sum()
            P_in_label = (frame_wise_label != -1).sum()
        else:
            target_cluster_id = cluster_to_id_mapper[target_cluster]
            TP = np.logical_and(frame_wise_label == target_cluster_id, frame_wise_prediction == frame_wise_label).sum()
            P_in_pred = (frame_wise_prediction == target_cluster_id).sum()
            P_in_label = (frame_wise_label == target_cluster_id).sum()

        precision = TP / max(P_in_pred, 1e-12)
        recall = TP / max(P_in_label, 1e-12)
        f1 = 2 / (1 / max(precision, 1e-12) + 1 / max(recall, 1e-12))

        return TP, P_in_pred, P_in_label, precision, recall, f1


class WhisperSegmenterForEval(SegmenterBase):
    def __init__(self, model_path=None, device=None, model=None, tokenizer=None):
        super().__init__()
        if model_path is not None:
            self.model, self.tokenizer = load_inference_model(model_path)
            if device is not None:
                self.model = self.model.to(device)
        else:
            try:
                self.model = model.module
            except:
                self.model = model

            self.tokenizer = tokenizer

        self.device = list(self.model.parameters())[0].device

        self.total_spec_columns = self.model.config.total_spec_columns
        self.cluster_codebook = self.model.config.cluster_codebook
        self.inverse_cluster_codebook = {cluster_id: cluster for cluster, cluster_id in self.cluster_codebook.items()}

        default_segmentation_config = getattr(self.model.config, "default_segmentation_config", None) or {}
        self.default_segmentation_config.update(default_segmentation_config)

    def generate_segment_records(
        self,
        audio,
        sr,
        min_frequency,
        spec_time_step,
        num_trials,
        batch_size,
        max_length,
        num_beams,
        top_k,
        top_p,
        length_penalty,
        status_monitor,
        max_frequency=None,
    ):
        generated_text_list = []
        sliced_audio_metadata = []
        batch = []

        for item in self.iter_sliced_audio_features(
            audio, sr, min_frequency, spec_time_step, num_trials, max_frequency=max_frequency
        ):
            batch.append(item)
            if len(batch) < batch_size:
                continue
            generated_text_list.extend(
                self.generate_segment_text(
                    batch, batch_size, max_length, num_beams, top_k, top_p, length_penalty, status_monitor
                )
            )
            sliced_audio_metadata.extend(
                (trial_id, offset_time, None, duration) for trial_id, offset_time, _, duration in batch
            )
            batch.clear()

        if batch:
            generated_text_list.extend(
                self.generate_segment_text(
                    batch, batch_size, max_length, num_beams, top_k, top_p, length_penalty, status_monitor
                )
            )
            sliced_audio_metadata.extend(
                (trial_id, offset_time, None, duration) for trial_id, offset_time, _, duration in batch
            )

        return generated_text_list, sliced_audio_metadata

    def generate_segment_text(
        self,
        sliced_audios_features,
        batch_size,
        max_length,
        num_beams,
        top_k=1,
        top_p=1.0,
        length_penalty=1.0,
        status_monitor=None,
    ):
        generated_text_list = []

        for pos in range(0, len(sliced_audios_features), batch_size):
            input_features = torch.from_numpy(
                np.asarray([item[2] for item in sliced_audios_features[pos : pos + batch_size]])
            ).to(device=self.device, dtype=next(self.model.parameters()).dtype)
            generated_ids = self.model.generate(
                input_features=input_features,
                decoder_input_ids=torch.LongTensor(
                    [
                        self.tokenizer.convert_tokens_to_ids(["<|startoftranscript|>", "<|en|>", "<|notimestamps|>"])
                        for _ in range(input_features.size(0))
                    ]
                ).to(self.device),
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                max_length=max_length,
                force_unique_generate_call=True,
                num_beams=num_beams,
                do_sample=num_beams == 1 and top_k > 1,
                top_k=top_k,
                top_p=top_p,
                length_penalty=length_penalty,
            )
            generated_text_batch = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
            generated_text_list += generated_text_batch
            del input_features, generated_ids
        return generated_text_list


class WhisperSegmenter(SegmenterBase):
    generate_segment_records = WhisperSegmenterForEval.generate_segment_records

    def __init__(
        self,
        model_path,
        device=None,
    ):
        super().__init__()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

        if device == "cuda" or device == "mps":
            self.device_list = [torch.device(device)]
            self.model_list = []
            self.tokenizer_list = []
            for model_device in self.device_list:
                model, tokenizer = load_inference_model(model_path)
                self.model_list.append(model.to(model_device))
                self.tokenizer_list.append(tokenizer)
        else:
            self.device_list = [torch.device(device)]
            model, tokenizer = load_inference_model(model_path)
            self.model_list = [model]
            self.tokenizer_list = [tokenizer]

        self.total_spec_columns = self.model_list[0].config.total_spec_columns
        self.cluster_codebook = self.model_list[0].config.cluster_codebook
        self.inverse_cluster_codebook = {cluster_id: cluster for cluster, cluster_id in self.cluster_codebook.items()}

        default_segmentation_config = getattr(self.model_list[0].config, "default_segmentation_config", None) or {}
        self.default_segmentation_config.update(default_segmentation_config)

    def generate_segment_text_core(
        self,
        sliced_audios_features,
        batch_size,
        max_length,
        num_beams,
        top_k,
        top_p,
        length_penalty,
        generated_texts_dict,
        thread_id,
        status_monitor=None,
    ):
        device = self.device_list[thread_id]
        model = self.model_list[thread_id]
        tokenizer = self.tokenizer_list[thread_id]
        generated_text_list = []
        for pos in range(0, len(sliced_audios_features), batch_size):
            input_features = torch.from_numpy(
                np.asarray([item[2] for item in sliced_audios_features[pos : pos + batch_size]])
            ).to(device=device, dtype=next(model.parameters()).dtype)
            generated_ids = model.generate(
                input_features=input_features,
                decoder_input_ids=torch.LongTensor(
                    [
                        tokenizer.convert_tokens_to_ids(["<|startoftranscript|>", "<|en|>", "<|notimestamps|>"])
                        for _ in range(input_features.size(0))
                    ]
                ).to(device),
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                max_length=max_length,
                force_unique_generate_call=True,
                num_beams=num_beams,
                do_sample=num_beams == 1,
                top_k=top_k,
                top_p=top_p,
                length_penalty=length_penalty,
            )
            generated_text_batch = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
            generated_text_list += generated_text_batch

            ### if status_monitor is not None, update the progress in percentage
            ### status_monitor is a dictionary which contains the key "progress"
            if status_monitor is not None:
                progress_percent = int(100 * min(1, (pos + batch_size) / len(sliced_audios_features)))
                status_monitor["progress"] = progress_percent

        generated_texts_dict[thread_id] = generated_text_list
