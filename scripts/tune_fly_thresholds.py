import argparse
import json
from pathlib import Path

import numpy as np

from das import api, prediction_results
from das.config import Config, load_config_mapping


def _score(cached_entries, context, **parameters):
    f1, wer = prediction_results._syllable_scores_from_cached_entries(
        cached_entries=cached_entries,
        class_names=context.class_names,
        class_types=context.class_types,
        frame_rate_hz=context.frame_rate_hz,
        fill_gap_seconds=parameters.get("fill_gap_ms", 0.0) / 1000.0,
        min_syllable_seconds=parameters.get("min_syllable_ms", 0.0) / 1000.0,
        tolerance_seconds=context.tolerance_seconds,
        postprocessor=context.postprocessor,
        segment_threshold_low=parameters.get("segment_threshold_low", 0.5),
        segment_threshold_high=parameters.get("segment_threshold_high", 0.5),
        event_threshold=parameters.get("event_threshold", 0.5),
    )
    return {**parameters, "f1": f1, "wer": wer}


def _best(results):
    return min(results, key=lambda result: (-result["f1"], result["wer"], tuple(result.values())))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=("pulse", "sine"))
    parser.add_argument("config")
    parser.add_argument("checkpoint")
    parser.add_argument("output")
    parser.add_argument("--file", action="append", default=[])
    args = parser.parse_args()

    config = Config.from_mapping(load_config_mapping(args.config), base=Config()).copy(
        mode="predict", checkpoint=args.checkpoint
    )
    api._seed_if_requested(config, verbose=False)
    datamodule, _, _ = api._build_train_datamodule(config.copy(mode="train"))
    if args.file:
        wanted = set(args.file)
        subset = datamodule.subsets["val"]
        indices = [index for index, audio_file in enumerate(subset["audio"]) if audio_file.name in wanted]
        datamodule.subsets["val"] = {key: [values[index] for index in indices] for key, values in subset.items()}
    loader = api._load_stage_dataloader(datamodule, "val", shuffle=False)
    model, runtime = api._load_predict_model(args.checkpoint, config=config)
    predictions = api._build_inference_trainer(config).predict(model, dataloaders=loader)
    probabilities = prediction_results._stitch_chunk_probabilities_by_file(
        loader.dataset,
        prediction_results._prediction_batches_to_chunk_probabilities(predictions),
    )
    entries, summary = prediction_results._annotated_probability_entries(
        dataset=loader.dataset,
        file_probabilities=probabilities,
    )
    cached_entries = [
        (prediction_results.load_reference_syllables(annotation_file=annotation, class_names=runtime.class_names), values)
        for _, annotation, values in entries
    ]
    context = api._prediction_result_context(config, runtime)

    if args.kind == "pulse":
        candidates = [_score(cached_entries, context, event_threshold=float(value)) for value in np.arange(0.05, 1.0, 0.05)]
        best = _best(candidates)
    else:
        thresholds = [
            _score(
                cached_entries,
                context,
                fill_gap_ms=20.0,
                min_syllable_ms=100.0,
                segment_threshold_low=low,
                segment_threshold_high=high,
            )
            for low in np.arange(0.05, 0.5, 0.05)
            for high in np.arange(0.5, 1.0, 0.05)
            if low <= high
        ]
        threshold_best = _best(thresholds)
        durations = [
            _score(
                cached_entries,
                context,
                fill_gap_ms=fill_gap_ms,
                min_syllable_ms=min_syllable_ms,
                segment_threshold_low=threshold_best["segment_threshold_low"],
                segment_threshold_high=threshold_best["segment_threshold_high"],
            )
            for fill_gap_ms in range(0, 51, 5)
            for min_syllable_ms in range(20, 201, 20)
        ]
        best = _best(durations)
        candidates = thresholds + durations

    report = {
        "kind": args.kind,
        "split": "val",
        "checkpoint": args.checkpoint,
        "evaluated_file_count": summary["evaluated_file_count"],
        "evaluated_window_count": len(loader),
        "best": best,
        "top_five": sorted(candidates, key=lambda result: (-result["f1"], result["wer"]))[:5],
    }
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
