"""
Phase 4.3 — Produce 3-8s training clips from phase 4.2 segments.

Rules:
  - Clips already 3-8s  → copy as-is
  - Clips > 8s          → split into equal sub-clips targeting ~5s each
  - Clips < 3s          → discard

Output: data/clips/{channel}/{speaker}/clip_{idx:06d}.wav
  (flat per-speaker directory, ready for XTTS fine-tuning dataset)

Usage:
    python -m pipeline.clip                    # all channels
    python -m pipeline.clip dahih              # single channel
"""

import sys
import math
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
SEGMENTS_DIR = BACKEND_DIR / "data" / "segments"
CLIPS_DIR = BACKEND_DIR / "data" / "clips"

MIN_DURATION = 3.0
MAX_DURATION = 8.0
TARGET_DURATION = 5.0  # aim for when splitting long clips


def split_audio(audio, sr: int, duration: float) -> list:
    """Split audio array into equal sub-clips targeting TARGET_DURATION each."""
    n_pieces = math.ceil(duration / TARGET_DURATION)
    samples_per_piece = len(audio) // n_pieces
    pieces = []
    for i in range(n_pieces):
        start = i * samples_per_piece
        end = start + samples_per_piece if i < n_pieces - 1 else len(audio)
        pieces.append(audio[start:end])
    return pieces


def process_channel(channel_dir: Path, out_base: Path) -> dict:
    import soundfile as sf

    channel = channel_dir.name
    log.info(f"Channel: {channel}")

    # Collect per-speaker counters across all source files
    speaker_counters: dict[str, int] = {}
    total_kept = total_split = total_dropped = 0

    for file_dir in sorted(channel_dir.iterdir()):
        if not file_dir.is_dir():
            continue

        wav_files = sorted(file_dir.glob("*.wav"))
        if not wav_files:
            continue

        kept = split_out = dropped = 0

        for wav_path in wav_files:
            # Speaker label is the prefix before the last underscore+digits
            # e.g. SPEAKER_04_0023.wav → SPEAKER_04
            parts = wav_path.stem.rsplit("_", 1)
            speaker = parts[0] if len(parts) == 2 else wav_path.stem

            audio, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
            duration = len(audio) / sr

            out_dir = out_base / channel / speaker
            out_dir.mkdir(parents=True, exist_ok=True)

            def next_idx(spk):
                idx = speaker_counters.get(spk, 0)
                speaker_counters[spk] = idx + 1
                return idx

            if duration < MIN_DURATION:
                dropped += 1
                continue

            if duration <= MAX_DURATION:
                idx = next_idx(speaker)
                sf.write(str(out_dir / f"clip_{idx:06d}.wav"), audio, sr, subtype="PCM_16")
                kept += 1
            else:
                pieces = split_audio(audio, sr, duration)
                for piece in pieces:
                    piece_dur = len(piece) / sr
                    if piece_dur < MIN_DURATION:
                        dropped += 1
                        continue
                    idx = next_idx(speaker)
                    sf.write(str(out_dir / f"clip_{idx:06d}.wav"), piece, sr, subtype="PCM_16")
                    split_out += 1

        total_kept += kept
        total_split += split_out
        total_dropped += dropped

    # Count total clips per speaker
    speaker_totals = {spk: cnt for spk, cnt in sorted(speaker_counters.items())}
    log.info(
        f"  kept={total_kept}  from_splits={total_split}  dropped={total_dropped}"
    )
    log.info(f"  clips per speaker: {speaker_totals}")

    return {
        "channel": channel,
        "kept": total_kept,
        "from_splits": total_split,
        "dropped": total_dropped,
        "speakers": speaker_totals,
    }


def main():
    channel_filter = set(sys.argv[1:]) if len(sys.argv) > 1 else None

    channels = [
        d for d in sorted(SEGMENTS_DIR.iterdir())
        if d.is_dir() and (not channel_filter or d.name in channel_filter)
    ]

    if not channels:
        log.error(f"No channels found under {SEGMENTS_DIR}")
        sys.exit(1)

    log.info(f"Processing {len(channels)} channel(s): {[c.name for c in channels]}")

    results = []
    for channel_dir in channels:
        result = process_channel(channel_dir, CLIPS_DIR)
        results.append(result)

    total_clips = sum(r["kept"] + r["from_splits"] for r in results)
    log.info("=" * 60)
    log.info("Clipping complete")
    log.info(f"  Total training clips: {total_clips}")
    log.info(f"  Output dir         : {CLIPS_DIR}")


if __name__ == "__main__":
    main()
