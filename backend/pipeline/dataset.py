"""
Phase 5.3 — Build master dataset CSV from clips + transcripts + prosody clusters.

Matches each clip back to the whisper transcript segment that best covers it,
using the diarization turn timestamps stored in the diarization JSON files.

Output:
  data/dataset.csv   — master manifest, ready for XTTS fine-tuning
  data/dataset_train.csv
  data/dataset_val.csv

Usage:
    python -m pipeline.dataset
"""

import json
import logging
import random
from pathlib import Path

import utils.audio_compat  # noqa: F401

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

BACKEND_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BACKEND_DIR / "data"
CLIPS_FINAL_DIR = DATA_DIR / "clips_final"
DIARIZE_DIR = DATA_DIR / "diarization"
TRANSCRIPT_DIR = DATA_DIR / "transcripts"
SPEAKER_MAP_PATH = DATA_DIR / "speaker_map.json"
PROSODY_CSV = DATA_DIR / "prosody_features.csv"
STYLE_LABELS_PATH = DATA_DIR / "style_labels.json"


def build_transcript_index() -> dict[str, list[dict]]:
    """Build {file_stem -> [whisper segments]} from all transcript JSONs."""
    index = {}
    for json_path in TRANSCRIPT_DIR.rglob("*.json"):
        data = json.loads(json_path.read_text(encoding="utf-8"))
        stem = json_path.stem
        index[stem] = data.get("segments", [])
    return index


def get_transcript_for_window(segments: list[dict], start: float, end: float) -> str:
    """Extract transcript text for a time window using whisper word timestamps."""
    words = []
    for seg in segments:
        for w in seg.get("words", []):
            ws, we = w.get("start", 0), w.get("end", 0)
            # Include word if it overlaps meaningfully with the window
            overlap_start = max(ws, start)
            overlap_end = min(we, end)
            if overlap_end - overlap_start > 0.05:
                words.append(w["word"])
    return " ".join(words).strip()


def build_turn_map(speaker_map: dict) -> dict:
    """
    Build a mapping from (narrator_id, clip_idx) → (source_file_stem, start, end)
    by replaying the reorganize logic against diarization JSONs.

    Returns: {narrator_id: [(file_stem, speaker_label, start, end), ...]}
    in the same order clips were written by reorganize.py.
    """
    turn_map = {}

    for channel, narrators in speaker_map["channels"].items():
        channel_diarize_dir = DIARIZE_DIR / channel
        for narrator_id, cfg in narrators.items():
            if narrator_id.startswith("_"):
                continue

            sources = cfg["sources"]
            skip_first_n = cfg.get("skip_first_n", 0)

            # Collect all (file_stem, speaker, start, end) in the same order
            # as reorganize.py iterated clips
            entries = []

            # reorganize.py iterates: sorted(channel_src / spk / *.wav)
            # We mirror that but use diarization turns instead
            for spk in sources:
                # Gather all turns for this speaker across all diarization files
                spk_entries = []
                for json_path in sorted(channel_diarize_dir.glob("*.json")):
                    data = json.loads(json_path.read_text(encoding="utf-8"))
                    for turn in data["turns"]:
                        if turn["speaker"] != spk:
                            continue
                        dur = turn["end"] - turn["start"]
                        if dur < 2.0 or dur > 30.0:
                            continue
                        spk_entries.append((json_path.stem, spk, turn["start"], turn["end"]))
                # Sort matches what reorganize.py did (sorted wav files = sorted by name)
                spk_entries.sort(key=lambda x: (x[0], x[2]))
                entries.extend(spk_entries)

            if skip_first_n > 0 and len(sources) == 1:
                entries = entries[skip_first_n:]

            # Now split long entries into sub-clips (mirrors clip.py logic)
            import math
            MIN_DUR, MAX_DUR, TARGET = 3.0, 8.0, 5.0
            final_entries = []
            for file_stem, spk, start, end in entries:
                duration = end - start
                if duration < MIN_DUR:
                    continue
                if duration <= MAX_DUR:
                    final_entries.append((file_stem, spk, start, end))
                else:
                    n_pieces = math.ceil(duration / TARGET)
                    piece_dur = duration / n_pieces
                    for i in range(n_pieces):
                        ps = start + i * piece_dur
                        pe = start + (i + 1) * piece_dur if i < n_pieces - 1 else end
                        if pe - ps >= MIN_DUR:
                            final_entries.append((file_stem, spk, ps, pe))

            turn_map[narrator_id] = final_entries

    return turn_map


def main():
    import pandas as pd

    style_labels = json.loads(STYLE_LABELS_PATH.read_text(encoding="utf-8"))
    speaker_map = json.loads(SPEAKER_MAP_PATH.read_text(encoding="utf-8"))
    prosody_df = pd.read_csv(PROSODY_CSV)

    # cluster → style tag lookup
    cluster_to_style = {int(k): v for k, v in style_labels.items() if k.isdigit()}

    log.info("Building transcript index...")
    transcript_index = build_transcript_index()
    log.info(f"  {len(transcript_index)} transcript files indexed")

    log.info("Building turn map...")
    turn_map = build_turn_map(speaker_map)
    for nid, entries in turn_map.items():
        log.info(f"  {nid}: {len(entries)} entries")

    # Build prosody lookup: clip_path → cluster
    prosody_lookup = {}
    for _, row in prosody_df.iterrows():
        prosody_lookup[row["clip_path"]] = int(row["cluster"])

    log.info("Assembling dataset rows...")
    rows = []
    missing_transcript = 0

    for narrator_id, entries in turn_map.items():
        narrator_dir = CLIPS_FINAL_DIR / narrator_id
        for idx, (file_stem, spk, start, end) in enumerate(entries):
            clip_path = narrator_dir / f"clip_{idx:06d}.wav"
            if not clip_path.exists():
                continue

            # Transcript
            segments = transcript_index.get(file_stem, [])
            transcript = get_transcript_for_window(segments, start, end)
            if not transcript:
                missing_transcript += 1

            # Style
            clip_rel = str(clip_path.relative_to(BACKEND_DIR))
            cluster = prosody_lookup.get(clip_rel, -1)
            style_tag = cluster_to_style.get(cluster, "unknown")

            rows.append({
                "clip_path": clip_rel,
                "narrator_id": narrator_id,
                "style_tag": style_tag,
                "cluster": cluster,
                "transcript": transcript,
                "source_file": file_stem,
                "start_s": round(start, 3),
                "end_s": round(end, 3),
                "duration_s": round(end - start, 3),
            })

    df = pd.DataFrame(rows)
    log.info(f"Total rows: {len(df)}  |  missing transcripts: {missing_transcript}")

    # Validation
    bad_dur = df[(df["duration_s"] < 3.0) | (df["duration_s"] > 8.0)]
    if len(bad_dur):
        log.warning(f"  {len(bad_dur)} clips outside 3-8s range (rounding artifact, will trim)")
        df = df[(df["duration_s"] >= 3.0) & (df["duration_s"] <= 8.0)]

    empty_tx = (df["transcript"] == "").sum()
    log.info(f"  Empty transcripts: {empty_tx} ({100*empty_tx/len(df):.1f}%)")

    # Distribution summary
    log.info("Narrator distribution:")
    for nid, cnt in df["narrator_id"].value_counts().items():
        log.info(f"  {nid:25s}: {cnt} clips")
    log.info("Style distribution:")
    for tag, cnt in df["style_tag"].value_counts().items():
        log.info(f"  {tag:20s}: {cnt} clips")

    # Train / val split (90/10, stratified by narrator)
    random.seed(42)
    train_rows, val_rows = [], []
    for nid, group in df.groupby("narrator_id"):
        indices = group.index.tolist()
        random.shuffle(indices)
        n_val = max(1, int(len(indices) * 0.1))
        val_rows.extend(indices[:n_val])
        train_rows.extend(indices[n_val:])

    df_train = df.loc[train_rows].reset_index(drop=True)
    df_val = df.loc[val_rows].reset_index(drop=True)

    # Save
    df.to_csv(DATA_DIR / "dataset.csv", index=False, encoding="utf-8")
    df_train.to_csv(DATA_DIR / "dataset_train.csv", index=False, encoding="utf-8")
    df_val.to_csv(DATA_DIR / "dataset_val.csv", index=False, encoding="utf-8")

    log.info("=" * 60)
    log.info(f"dataset.csv      : {len(df)} rows")
    log.info(f"dataset_train.csv: {len(df_train)} rows")
    log.info(f"dataset_val.csv  : {len(df_val)} rows")
    log.info(f"Output: {DATA_DIR}")


if __name__ == "__main__":
    main()
