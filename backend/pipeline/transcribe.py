"""
Phase 3.1 — Whisper transcription using faster-whisper large-v3.

Processes all WAV files in data/audio/{speaker}/ and outputs:
  - data/transcripts/{speaker}/{name}.json  (word-level timestamps)
  - data/transcripts/{speaker}/{name}.txt   (plain text)

Usage:
    python -m pipeline.transcribe              # all speakers
    python -m pipeline.transcribe dahih        # single speaker
"""

import sys
import json
import time
import logging
from pathlib import Path

# torchaudio patch must come first
import utils.audio_compat  # noqa: F401

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

BACKEND_DIR = Path(__file__).resolve().parent.parent
AUDIO_DIR = BACKEND_DIR / "data" / "audio"
TRANSCRIPT_DIR = BACKEND_DIR / "data" / "transcripts"

MODEL_SIZE = "large-v3"
LANGUAGE = "ar"
DEVICE = "cuda"
COMPUTE_TYPE = "float16"  # fits in ~1.5 GB on RTX 3050 6GB


def load_model():
    from faster_whisper import WhisperModel
    log.info(f"Loading faster-whisper {MODEL_SIZE} on {DEVICE} ({COMPUTE_TYPE})")
    model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
    log.info("Model loaded")
    return model


def transcribe_file(model, wav_path: Path, out_dir: Path) -> dict:
    """Transcribe a single WAV file. Returns a summary dict."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = wav_path.stem
    json_path = out_dir / f"{stem}.json"
    txt_path = out_dir / f"{stem}.txt"

    if json_path.exists() and txt_path.exists():
        log.info(f"  SKIP (already done): {stem}")
        return {"file": wav_path.name, "status": "skipped"}

    log.info(f"  Transcribing: {wav_path.name}")
    t0 = time.time()

    segments, info = model.transcribe(
        str(wav_path),
        language=LANGUAGE,
        word_timestamps=True,
        vad_filter=True,          # skip silent regions → faster + fewer hallucinations
        vad_parameters={"min_silence_duration_ms": 500},
    )

    # Consume the generator and build output structure
    result_segments = []
    full_text_parts = []

    for seg in segments:
        words = []
        if seg.words:
            for w in seg.words:
                words.append({
                    "word": w.word,
                    "start": round(w.start, 3),
                    "end": round(w.end, 3),
                    "probability": round(w.probability, 4),
                })
        result_segments.append({
            "id": seg.id,
            "start": round(seg.start, 3),
            "end": round(seg.end, 3),
            "text": seg.text.strip(),
            "words": words,
        })
        full_text_parts.append(seg.text.strip())

    elapsed = time.time() - t0
    audio_duration = info.duration
    rtf = elapsed / audio_duration if audio_duration > 0 else 0

    output = {
        "file": wav_path.name,
        "language": info.language,
        "language_probability": round(info.language_probability, 4),
        "duration_s": round(audio_duration, 2),
        "model": MODEL_SIZE,
        "segments": result_segments,
    }

    json_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    txt_path.write_text("\n".join(full_text_parts), encoding="utf-8")

    log.info(
        f"    Done in {elapsed:.0f}s | audio={audio_duration/60:.1f}min | "
        f"RTF={rtf:.2f}x | segments={len(result_segments)}"
    )

    return {
        "file": wav_path.name,
        "status": "done",
        "duration_s": audio_duration,
        "elapsed_s": elapsed,
        "segments": len(result_segments),
    }


def report_vram():
    try:
        import torch
        if torch.cuda.is_available():
            used = torch.cuda.memory_allocated() / 1024 ** 3
            reserved = torch.cuda.memory_reserved() / 1024 ** 3
            log.info(f"VRAM: {used:.2f} GB allocated / {reserved:.2f} GB reserved")
    except Exception:
        pass


def main():
    # Optional: filter to specific speaker(s) via CLI args
    speaker_filter = set(sys.argv[1:]) if len(sys.argv) > 1 else None

    # Collect WAV files
    wav_files: list[tuple[Path, Path]] = []  # (wav_path, out_dir)
    for speaker_dir in sorted(AUDIO_DIR.iterdir()):
        if not speaker_dir.is_dir():
            continue
        if speaker_filter and speaker_dir.name not in speaker_filter:
            continue
        out_dir = TRANSCRIPT_DIR / speaker_dir.name
        for wav in sorted(speaker_dir.glob("*.wav")):
            wav_files.append((wav, out_dir))

    if not wav_files:
        log.error(f"No WAV files found under {AUDIO_DIR}")
        sys.exit(1)

    total_duration = 0.0
    log.info(f"Found {len(wav_files)} WAV file(s) to transcribe")

    model = load_model()
    report_vram()

    results = []
    for wav_path, out_dir in wav_files:
        result = transcribe_file(model, wav_path, out_dir)
        results.append(result)
        if result.get("duration_s"):
            total_duration += result["duration_s"]

    # Summary
    done = [r for r in results if r["status"] == "done"]
    skipped = [r for r in results if r["status"] == "skipped"]
    total_elapsed = sum(r.get("elapsed_s", 0) for r in done)

    log.info("=" * 60)
    log.info(f"Transcription complete")
    log.info(f"  Files processed : {len(done)} done, {len(skipped)} skipped")
    log.info(f"  Total audio     : {total_duration/3600:.2f}h")
    log.info(f"  Total wall time : {total_elapsed/60:.1f} min")
    if total_elapsed > 0 and total_duration > 0:
        log.info(f"  Overall RTF     : {total_elapsed/total_duration:.2f}x")
    log.info(f"  Output dir      : {TRANSCRIPT_DIR}")

    # Unload model to free VRAM
    del model
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    log.info("Model unloaded, VRAM freed")


if __name__ == "__main__":
    main()
