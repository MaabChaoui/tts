"""
Phase 4.1 — Speaker diarization using pyannote.audio.

Processes all WAV files in data/audio/{speaker}/ and outputs:
  - data/diarization/{speaker}/{name}.rttm  (standard RTTM format)
  - data/diarization/{speaker}/{name}.json  (simplified turn list)

Usage:
    python -m pipeline.diarize              # all speakers
    python -m pipeline.diarize dahih        # single speaker
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
DIARIZE_DIR = BACKEND_DIR / "data" / "diarization"


def load_pipeline():
    import torch
    from pyannote.audio import Pipeline
    from dotenv import load_dotenv
    import os

    load_dotenv(BACKEND_DIR / ".env")
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN not set in backend/.env")

    log.info("Loading pyannote speaker-diarization-3.1")
    pipe = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=token,
    )
    pipe.to(torch.device("cuda"))
    log.info("Pipeline loaded")
    return pipe


def diarize_file(pipe, wav_path: Path, out_dir: Path) -> dict:
    """Diarize a single WAV file. Returns a summary dict."""
    import soundfile as sf
    import torch

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = wav_path.stem
    rttm_path = out_dir / f"{stem}.rttm"
    json_path = out_dir / f"{stem}.json"

    if rttm_path.exists() and json_path.exists():
        log.info(f"  SKIP (already done): {stem}")
        return {"file": wav_path.name, "status": "skipped"}

    log.info(f"  Diarizing: {wav_path.name}")
    t0 = time.time()

    # Load audio as waveform dict — never pass file path to pyannote
    waveform, sample_rate = sf.read(str(wav_path), dtype="float32", always_2d=True)
    waveform = torch.from_numpy(waveform.T)  # (channels, samples)
    audio_input = {"waveform": waveform, "sample_rate": sample_rate}
    audio_duration = waveform.shape[1] / sample_rate

    diarization = pipe(audio_input, batch_size=64)

    # pyannote 4.x returns DiarizeOutput dataclass; extract the Annotation
    annotation = diarization.speaker_diarization if hasattr(diarization, "speaker_diarization") else diarization

    # Write RTTM
    with open(rttm_path, "w", encoding="utf-8") as f:
        annotation.write_rttm(f)

    # Build simplified turn list
    turns = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        turns.append({
            "start": round(turn.start, 3),
            "end": round(turn.end, 3),
            "speaker": speaker,
        })

    output = {
        "file": wav_path.name,
        "duration_s": round(audio_duration, 2),
        "num_speakers": len({t["speaker"] for t in turns}),
        "num_turns": len(turns),
        "turns": turns,
    }
    json_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    elapsed = time.time() - t0
    rtf = elapsed / audio_duration if audio_duration > 0 else 0
    log.info(
        f"    Done in {elapsed:.0f}s | audio={audio_duration/60:.1f}min | "
        f"RTF={rtf:.2f}x | speakers={output['num_speakers']} | turns={len(turns)}"
    )

    return {
        "file": wav_path.name,
        "status": "done",
        "duration_s": audio_duration,
        "elapsed_s": elapsed,
        "num_speakers": output["num_speakers"],
        "num_turns": len(turns),
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
    speaker_filter = set(sys.argv[1:]) if len(sys.argv) > 1 else None

    wav_files: list[tuple[Path, Path]] = []
    for speaker_dir in sorted(AUDIO_DIR.iterdir()):
        if not speaker_dir.is_dir():
            continue
        if speaker_filter and speaker_dir.name not in speaker_filter:
            continue
        out_dir = DIARIZE_DIR / speaker_dir.name
        for wav in sorted(speaker_dir.glob("*.wav")):
            wav_files.append((wav, out_dir))

    if not wav_files:
        log.error(f"No WAV files found under {AUDIO_DIR}")
        sys.exit(1)

    log.info(f"Found {len(wav_files)} WAV file(s) to diarize")

    pipe = load_pipeline()
    report_vram()

    results = []
    total_duration = 0.0
    for wav_path, out_dir in wav_files:
        result = diarize_file(pipe, wav_path, out_dir)
        results.append(result)
        if result.get("duration_s"):
            total_duration += result["duration_s"]
        report_vram()

    done = [r for r in results if r["status"] == "done"]
    skipped = [r for r in results if r["status"] == "skipped"]
    total_elapsed = sum(r.get("elapsed_s", 0) for r in done)

    log.info("=" * 60)
    log.info("Diarization complete")
    log.info(f"  Files processed : {len(done)} done, {len(skipped)} skipped")
    log.info(f"  Total audio     : {total_duration/3600:.2f}h")
    log.info(f"  Total wall time : {total_elapsed/60:.1f} min")
    if total_elapsed > 0 and total_duration > 0:
        log.info(f"  Overall RTF     : {total_elapsed/total_duration:.2f}x")
    log.info(f"  Output dir      : {DIARIZE_DIR}")

    del pipe
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    log.info("Pipeline unloaded, VRAM freed")


if __name__ == "__main__":
    main()
