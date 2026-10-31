"""
Phase 4.2 — Extract per-speaker audio clips from diarization output.

Reads diarization JSON + source WAV, extracts each speaker turn with
duration in [MIN_DURATION, MAX_DURATION] seconds, and saves to:
  data/segments/{channel}/{file_stem}/{speaker}_{idx:04d}.wav

Usage:
    python -m pipeline.segment                    # all speakers
    python -m pipeline.segment dahih              # single channel
    python -m pipeline.segment dahih NatGeoHistory  # multiple
"""

import sys
import json
import logging
from pathlib import Path

import utils.audio_compat  # noqa: F401

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

BACKEND_DIR = Path(__file__).resolve().parent.parent
AUDIO_DIR = BACKEND_DIR / "data" / "audio"
DIARIZE_DIR = BACKEND_DIR / "data" / "diarization"
SEGMENTS_DIR = BACKEND_DIR / "data" / "segments"

MIN_DURATION = 2.0   # seconds — shorter turns are too noisy for training
MAX_DURATION = 30.0  # seconds — longer turns will be handled in phase 4.3


def process_file(json_path: Path, wav_path: Path, out_dir: Path) -> dict:
    import soundfile as sf
    import numpy as np

    stem = wav_path.stem
    file_out_dir = out_dir / stem

    # Skip if already done (check marker file)
    marker = file_out_dir / ".done"
    if marker.exists():
        log.info(f"  SKIP (already done): {stem}")
        return {"file": wav_path.name, "status": "skipped"}

    log.info(f"  Segmenting: {wav_path.name}")

    diarization = json.load(json_path.open(encoding="utf-8"))
    turns = diarization["turns"]

    # Load full audio once
    audio, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)

    file_out_dir.mkdir(parents=True, exist_ok=True)

    # Track per-speaker clip index
    speaker_idx: dict[str, int] = {}
    kept = 0
    skipped_short = 0
    skipped_long = 0

    for turn in turns:
        start = turn["start"]
        end = turn["end"]
        speaker = turn["speaker"]
        duration = end - start

        if duration < MIN_DURATION:
            skipped_short += 1
            continue
        if duration > MAX_DURATION:
            skipped_long += 1
            continue

        idx = speaker_idx.get(speaker, 0)
        speaker_idx[speaker] = idx + 1

        clip_path = file_out_dir / f"{speaker}_{idx:04d}.wav"
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        clip = audio[start_sample:end_sample]

        sf.write(str(clip_path), clip, sr, subtype="PCM_16")
        kept += 1

    # Write marker so we can skip on re-run
    marker.write_text("done")

    # Per-speaker summary
    speaker_summary = {spk: cnt for spk, cnt in speaker_idx.items()}

    log.info(
        f"    kept={kept}  short_skip={skipped_short}  long_skip={skipped_long}  "
        f"speakers={speaker_summary}"
    )

    return {
        "file": wav_path.name,
        "status": "done",
        "kept": kept,
        "skipped_short": skipped_short,
        "skipped_long": skipped_long,
        "speakers": speaker_summary,
    }


def main():
    channel_filter = set(sys.argv[1:]) if len(sys.argv) > 1 else None

    tasks: list[tuple[Path, Path, Path]] = []  # (json, wav, out_dir)
    for channel_dir in sorted(DIARIZE_DIR.iterdir()):
        if not channel_dir.is_dir():
            continue
        if channel_filter and channel_dir.name not in channel_filter:
            continue
        wav_dir = AUDIO_DIR / channel_dir.name
        out_dir = SEGMENTS_DIR / channel_dir.name
        for json_path in sorted(channel_dir.glob("*.json")):
            wav_path = wav_dir / (json_path.stem + ".wav")
            if not wav_path.exists():
                log.warning(f"  WAV not found, skipping: {wav_path}")
                continue
            tasks.append((json_path, wav_path, out_dir))

    if not tasks:
        log.error("No tasks found")
        sys.exit(1)

    log.info(f"Found {len(tasks)} file(s) to segment")

    total_kept = 0
    results = []
    for json_path, wav_path, out_dir in tasks:
        result = process_file(json_path, wav_path, out_dir)
        results.append(result)
        if result["status"] == "done":
            total_kept += result["kept"]

    done = [r for r in results if r["status"] == "done"]
    skipped = [r for r in results if r["status"] == "skipped"]

    log.info("=" * 60)
    log.info("Segmentation complete")
    log.info(f"  Files processed : {len(done)} done, {len(skipped)} skipped")
    log.info(f"  Total clips kept: {total_kept}")
    log.info(f"  Output dir      : {SEGMENTS_DIR}")


if __name__ == "__main__":
    main()
