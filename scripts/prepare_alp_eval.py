"""Prepare small, non-overlapping Meerkat and InfantMarmosetsVox benchmarks."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
from io import BytesIO, StringIO
import json
from pathlib import Path
import random
import shutil
import tarfile

import fsspec
import numpy as np
import pandas as pd
import requests
import soundfile as sf


BUCKET = "https://storage.googleapis.com/esp-data-274503/"
MARM_DOC = "https://zenodo.org/api/records/10130104/files/InfantMarmosetsVox_documentation.tar.gz/content"
CALL_TYPES = ["Peep(Pre-Phee)", "Phee", "Twitter", "Trill", "Trillphee", "TsikTse",
              "Egg", "Pheecry(cry)", "TrllTwitter", "Pheetwitter", "Peep"]
SEED = 42
CLIP_SECONDS = 30


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(url, path):
    if not path.exists():
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        path.write_bytes(response.content)
    return path.read_bytes()


def candidates(events, duration, stride=CLIP_SECONDS):
    """Reject whole windows containing overlaps or a boundary-cut vocalization."""
    events = events.sort_values(["start", "end"])
    assert np.isfinite(events[["start", "end"]]).all().all()
    assert ((events.start >= 0) & (events.end >= events.start) & (events.end <= duration)).all()
    clips = []
    for start in range(0, int(duration) - CLIP_SECONDS + 1, stride):
        end = start + CLIP_SECONDS
        rows = events.loc[(events.end > start) & (events.start < end)]
        if ((events.start == events.end) & (events.start >= start) & (events.start <= end)).any():
            continue  # Exclude windows with a zero-duration source annotation.
        if ((rows.start < start) | (rows.end > end)).any():
            continue
        if np.any(rows.start.to_numpy()[1:] < rows.end.cummax().to_numpy()[:-1]):
            continue
        clips.append({"start": start, "end": end, "events": rows.to_dict("records")})
    return clips


def select(clips, count, rng, *, max_per_source=None):
    """Cover available call types, then sample uniformly; keep a negative if possible."""
    pool = list(clips)
    rng.shuffle(pool)
    chosen, covered, sources = [], set(), Counter()
    for _ in range(count):
        eligible = [c for c in pool if max_per_source is None or sources[c["source"]] < max_per_source]
        if not eligible:
            raise ValueError(f"Only {len(chosen)} eligible clips; need {count}.")
        if not chosen and any(not c["events"] for c in eligible):
            item = next(c for c in eligible if not c["events"])
        else:
            item = max(eligible, key=lambda c: len({e["label"] for e in c["events"]} - covered))
        chosen.append(item)
        covered.update(e["label"] for e in item["events"])
        sources[item["source"]] += 1
        pool.remove(item)
        if "start" in item:
            pool = [c for c in pool if c["source"] != item["source"] or
                    c["end"] <= item["start"] or c["start"] >= item["end"]]
    return chosen


def make_plan(cache):
    rng = random.Random(SEED)
    selected, metadata = [], []
    meerkat_root = BUCKET + "voxaboxen/files/MT/formatted/"
    for split, count in [("train", 30), ("val", 10), ("test", 10)]:
        path = cache / f"{split}_info.csv"
        url = meerkat_root + path.name
        table = pd.read_csv(BytesIO(fetch(url, path)))
        metadata.append({"url": url, "file": path.name, "sha256": sha256(path)})
        for row in table.itertuples():
            events = pd.read_csv(StringIO(row.selection_table_str), sep="\t").rename(
                columns={"Begin Time (s)": "start", "End Time (s)": "end", "Annotation": "label"})
            assert set(events.label) == {"voc"}
            events["event_id"] = [f"{row.fn}:{i}" for i in range(len(events))]
            pool = candidates(events[["start", "end", "label", "event_id"]], row.audio_duration)
            for clip in pool:
                clip.update(dataset="meerkat", split=split, source=meerkat_root + row.audio_fp,
                            recording=row.fn, source_partition=f"MT_{split}")
            selected.extend(select(pool, count, rng))

    path = cache / "labels.csv"
    url = BUCKET + "infant_marmosets_vox/labels.csv"
    labels = pd.read_csv(BytesIO(fetch(url, path)))
    metadata.append({"url": url, "file": path.name, "sha256": sha256(path)})
    path = cache / "marmoset_documentation.tar.gz"
    archive = fetch(MARM_DOC, path)
    metadata.append({"url": MARM_DOC, "file": path.name, "sha256": sha256(path)})
    with tarfile.open(fileobj=BytesIO(archive)) as tar:
        raw = pd.read_csv(tar.extractfile("InfantMarmosetsVox/labels.csv"))
        (cache / "marmoset_README.txt").write_bytes(tar.extractfile("InfantMarmosetsVox/README.txt").read())
    # Raw silence/noise annotations establish the full annotated recording extent.
    duration = raw.groupby("filename").end.max()
    pool = []
    for recording, rows in labels.groupby("path", sort=True):
        stem = Path(recording).stem
        twin, caller = int(rows.twinID.iloc[0]), int(rows.callerID.iloc[0])
        assert rows.twinID.nunique() == rows.callerID.nunique() == 1
        assert set(rows.calltypeID).issubset(range(len(CALL_TYPES)))
        events = rows.rename(columns={"vocID": "event_id"}).copy()
        events["label"] = events.calltypeID.map(dict(enumerate(CALL_TYPES)))
        clips = candidates(events[["start", "end", "label", "event_id", "calltypeID"]], duration[stem], stride=5)
        for clip in clips:
            clip.update(dataset="infant_marmoset", split="train" if twin <= 3 else "val" if twin == 4 else "test",
                        source=BUCKET + "infant_marmosets_vox/" + recording,
                        recording=stem, caller_id=caller, twin_id=twin)
        pool.extend(clips)
    assert {c["caller_id"] for c in pool} == set(range(10))
    for caller in range(10):
        selected.extend(select([c for c in pool if c["caller_id"] == caller], 10, rng, max_per_source=2))
    for clip in selected:
        clip["file"] = f'{clip["dataset"]}/{clip["split"]}/{clip["recording"]}_{clip["start"]:06d}.wav'
    return {"seed": SEED, "clip_seconds": CLIP_SECONDS, "source_metadata": metadata, "clips": selected}


def extract(plan, output):
    def write_clip(item):
        i, clip = item
        target = output / clip["file"]
        if target.exists() and target.with_suffix(".csv").exists():
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        with fsspec.open(clip["source"], "rb", block_size=1024 * 1024) as remote:
            with sf.SoundFile(remote) as audio:
                assert audio.channels == 1
                rate = audio.samplerate
                audio.seek(clip["start"] * rate)
                samples = audio.read(CLIP_SECONDS * rate, dtype="float32")
        assert len(samples) == CLIP_SECONDS * rate and np.isfinite(samples).all()
        sf.write(target, samples, rate, subtype="FLOAT")
        rows = [{"name": event["label"], "start_seconds": event["start"] - clip["start"],
                 "stop_seconds": event["end"] - clip["start"]} for event in clip["events"]]
        pd.DataFrame(rows, columns=["name", "start_seconds", "stop_seconds"]).to_csv(
            target.with_suffix(".csv"), index=False, float_format="%.9f")
        print(f'{i + 1}/{len(plan["clips"])} {clip["file"]}: {len(rows)} calls, {rate} Hz', flush=True)

    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(write_clip, enumerate(plan["clips"])))


def verify(plan, output):
    counts, seen_events, intervals = {}, set(), {}
    for clip in plan["clips"]:
        path = output / clip["file"]
        audio, rate = sf.read(path, dtype="float32")
        assert audio.ndim == 1 and len(audio) == CLIP_SECONDS * rate and np.isfinite(audio).all()
        rows = pd.read_csv(path.with_suffix(".csv"))
        assert len(rows) == len(clip["events"])
        for row, event in zip(rows.itertuples(), clip["events"], strict=True):
            assert row.name == event["label"]
            assert abs(row.start_seconds + clip["start"] - event["start"]) < 1e-8
            assert abs(row.stop_seconds + clip["start"] - event["end"]) < 1e-8
            assert 0 <= row.start_seconds < row.stop_seconds <= CLIP_SECONDS
            event_key = (clip["dataset"], clip["source"], event["event_id"])
            assert event_key not in seen_events
            seen_events.add(event_key)
        assert not np.any(rows.start_seconds.to_numpy()[1:] < rows.stop_seconds.cummax().to_numpy()[:-1])
        for start, end in intervals.setdefault(clip["source"], []):
            assert clip["end"] <= start or clip["start"] >= end
        intervals[clip["source"]].append((clip["start"], clip["end"]))
        key = f'{clip["dataset"]}/{clip["split"]}'
        total = counts.setdefault(key, {"clips": 0, "seconds": 0, "events": 0, "negative_clips": 0,
                                        "labels": Counter(), "recordings": set(), "callers": set(),
                                        "twins": set(), "sample_rates": set()})
        total["clips"] += 1
        total["seconds"] += len(audio) / rate
        total["events"] += len(rows)
        total["negative_clips"] += int(rows.empty)
        total["labels"].update(rows.name)
        total["recordings"].add(clip["recording"])
        total["sample_rates"].add(rate)
        if "caller_id" in clip:
            total["callers"].add(clip["caller_id"])
            total["twins"].add(clip["twin_id"])
        clip["audio_sha256"] = sha256(path)
        clip["labels_sha256"] = sha256(path.with_suffix(".csv"))
    for dataset in ("meerkat", "infant_marmoset"):
        for split, seconds in [("train", 1800), ("val", 600), ("test", 600)]:
            assert counts[f"{dataset}/{split}"]["seconds"] == seconds
        groups = [counts[f"{dataset}/{s}"] for s in ("train", "val", "test")]
        for a, b in [(0, 1), (0, 2), (1, 2)]:
            for field in ("recordings", "callers", "twins"):
                assert groups[a][field].isdisjoint(groups[b][field])
        train_labels = set(groups[0]["labels"])
        assert set(groups[1]["labels"]) <= train_labels and set(groups[2]["labels"]) <= train_labels
    for total in counts.values():
        for field in ("recordings", "callers", "twins", "sample_rates"):
            total[field] = sorted(total[field])
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("dat/alp-eval"))
    parser.add_argument("--cache", type=Path, default=Path("scratch/alp-eval-cache"))
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    plan = make_plan(args.cache)
    args.output.mkdir(parents=True, exist_ok=True)
    plan_path = args.output / "plan.json"
    if plan_path.exists():
        assert json.loads(plan_path.read_text()) == plan, "Existing extraction plan differs; choose a new output directory."
    else:
        if any(args.output.iterdir()):
            raise FileExistsError(f"Output directory is not empty: {args.output}")
        plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    extract(plan, args.output)
    summary = verify(plan, args.output)
    (args.output / "manifest.json").write_text(json.dumps({**plan, "summary": summary}, indent=2) + "\n")
    sources = args.output / "sources"
    sources.mkdir(exist_ok=True)
    for source in plan["source_metadata"]:
        shutil.copyfile(args.cache / source["file"], sources / source["file"])
    shutil.copyfile(args.cache / "marmoset_README.txt", sources / "marmoset_README.txt")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
