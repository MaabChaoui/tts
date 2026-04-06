"""
Phase 2 — Audio Extraction & Normalization

Converts raw MP3/MP4 files to mono WAV at 24000 Hz (XTTS v2 native rate).
Preserves speaker directory structure: data/raw/{speaker}/ → data/audio/{speaker}/

Usage:
    python -m pipeline.extract
    python -m pipeline.extract --dry-run
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

# Project paths
BACKEND_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BACKEND_DIR / "data" / "raw"
AUDIO_DIR = BACKEND_DIR / "data" / "audio"

TARGET_SR = 24000  # XTTS v2 native sample rate
AUDIO_EXTENSIONS = {".mp3", ".mp4", ".m4a", ".wav", ".webm", ".ogg", ".flac"}


def sanitize_filename(name: str) -> str:
    """Convert Arabic/special filenames to safe ASCII-friendly names."""
    # Remove extension
    stem = Path(name).stem
    # Replace problematic chars with underscores
    safe = re.sub(r'[^\w\s\-.]', '_', stem, flags=re.UNICODE)
    # Collapse multiple underscores/spaces
    safe = re.sub(r'[\s_]+', '_', safe).strip('_')
    # Truncate to reasonable length
    if len(safe) > 120:
        safe = safe[:120]
    return safe


def get_duration(filepath: Path) -> float:
    """Get audio duration in seconds using ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "quiet",
            "-show_entries", "format=duration",
            "-of", "json",
            str(filepath),
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return 0.0
    info = json.loads(result.stdout)
    return float(info.get("format", {}).get("duration", 0))


def extract_audio(src: Path, dst: Path, dry_run: bool = False) -> float:
    """Convert a single file to mono WAV at TARGET_SR. Returns duration."""
    if dry_run:
        dur = get_duration(src)
        print(f"  [DRY RUN] {src.name} → {dst.name} ({dur:.1f}s)")
        return dur

    dst.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y", "-i", str(src),
        "-ac", "1",              # mono
        "-ar", str(TARGET_SR),   # 24000 Hz
        "-sample_fmt", "s16",    # 16-bit PCM
        "-vn",                   # no video
        str(dst),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR: {src.name}")
        print(f"    {result.stderr[-200:]}")
        return 0.0

    dur = get_duration(dst)
    return dur


def main():
    parser = argparse.ArgumentParser(description="Extract & normalize audio")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    args = parser.parse_args()

    if not RAW_DIR.exists():
        print(f"ERROR: {RAW_DIR} does not exist")
        sys.exit(1)

    # Discover all audio files in speaker subdirectories
    files = []
    for speaker_dir in sorted(RAW_DIR.iterdir()):
        if not speaker_dir.is_dir():
            continue
        speaker = speaker_dir.name
        for f in sorted(speaker_dir.iterdir()):
            if f.suffix.lower() in AUDIO_EXTENSIONS:
                safe_name = sanitize_filename(f.name)
                dst = AUDIO_DIR / speaker / f"{safe_name}.wav"
                files.append((f, dst, speaker))

    if not files:
        print("No audio files found in raw/ subdirectories.")
        sys.exit(1)

    print(f"Found {len(files)} files across {len(set(s for _, _, s in files))} speakers")
    print(f"Target: mono, {TARGET_SR} Hz, 16-bit PCM WAV")
    print(f"Output: {AUDIO_DIR}/")
    print()

    total_duration = 0.0
    speaker_durations = {}

    for src, dst, speaker in files:
        if dst.exists() and not args.dry_run:
            dur = get_duration(dst)
            print(f"  SKIP (exists) {dst.name} ({dur:.1f}s)")
        else:
            dur = extract_audio(src, dst, dry_run=args.dry_run)
            if not args.dry_run:
                print(f"  OK {dst.name} ({dur:.1f}s)")

        total_duration += dur
        speaker_durations[speaker] = speaker_durations.get(speaker, 0) + dur

    # Summary
    print(f"\n{'=' * 50}")
    print(f"  Total: {total_duration / 3600:.2f} hours ({len(files)} files)")
    for speaker, dur in sorted(speaker_durations.items()):
        h = dur / 3600
        print(f"  {speaker:20s}: {h:.2f}h ({dur / 60:.0f} min)")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
