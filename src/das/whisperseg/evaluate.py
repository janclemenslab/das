from tqdm import tqdm


def evaluate(audio_list, label_list, segmenter, batch_size, max_length, num_trials, num_beams=4, target_cluster=None):
    total_n_true_positive_segment_wise, total_n_positive_in_prediction_segment_wise, total_n_positive_in_label_segment_wise = 0, 0, 0
    total_n_true_positive_frame_wise, total_n_positive_in_prediction_frame_wise, total_n_positive_in_label_frame_wise = 0, 0, 0

    for audio, label in tqdm(zip(audio_list, label_list), total=len(audio_list)):
        prediction = segmenter.segment(
            audio,
            sr=label["sr"],
            min_frequency=label.get("min_frequency", None),
            max_frequency=label.get("max_frequency", None),
            spec_time_step=label.get("spec_time_step", None),
            max_length=max_length,
            batch_size=batch_size,
            num_trials=num_trials,
            num_beams=num_beams,
        )

        TP, P_pred, P_label = segmenter.segment_score(prediction, label, target_cluster=target_cluster)[:3]
        total_n_true_positive_segment_wise += TP
        total_n_positive_in_prediction_segment_wise += P_pred
        total_n_positive_in_label_segment_wise += P_label

        TP, P_pred, P_label = segmenter.frame_score(prediction, label, target_cluster=target_cluster)[:3]

        total_n_true_positive_frame_wise += TP
        total_n_positive_in_prediction_frame_wise += P_pred
        total_n_positive_in_label_frame_wise += P_label

    res = {}

    precision = total_n_true_positive_segment_wise / max(total_n_positive_in_prediction_segment_wise, 1e-12)
    recall = total_n_true_positive_segment_wise / max(total_n_positive_in_label_segment_wise, 1e-12)
    f1 = 2 / (1 / max(precision, 1e-12) + 1 / max(recall, 1e-12))

    res["segment_wise"] = [
        int(total_n_true_positive_segment_wise),
        int(total_n_positive_in_prediction_segment_wise),
        int(total_n_positive_in_label_segment_wise),
        float(precision),
        float(recall),
        float(f1),
    ]

    precision = total_n_true_positive_frame_wise / max(total_n_positive_in_prediction_frame_wise, 1e-12)
    recall = total_n_true_positive_frame_wise / max(total_n_positive_in_label_frame_wise, 1e-12)
    f1 = 2 / (1 / max(precision, 1e-12) + 1 / max(recall, 1e-12))

    res["frame_wise"] = [
        int(total_n_true_positive_frame_wise),
        int(total_n_positive_in_prediction_frame_wise),
        int(total_n_positive_in_label_frame_wise),
        float(precision),
        float(recall),
        float(f1),
    ]

    return res
