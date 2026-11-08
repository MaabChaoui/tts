"""
Zero-shot voice cloning demo with pretrained XTTS v2 (no fine-tuning).

Usage:
    .venv/bin/python clone_demo.py --ref outputs/ref/me.wav --text "..." --lang en

Writes a WAV and a JSON run record to outputs/clone/. outputs/ is gitignored;
reference recordings are personal voice data and must stay out of Git.
Use only voices whose owners consented (e.g. your own).
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

import soundfile as sf
import torch

sys.path.insert(0, os.path.dirname(__file__))
import utils.audio_compat  # noqa: F401 — patches torchaudio.load/save

os.environ["COQUI_TOS_AGREED"] = "1"

MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"
HERE = os.path.dirname(os.path.abspath(__file__))


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", HERE, "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", required=True, help="reference WAV of the consenting speaker")
    parser.add_argument("--text", required=True)
    parser.add_argument("--lang", default="en", help="XTTS language code, e.g. en, fr, ar, es, ru")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default=os.path.join(HERE, "outputs", "clone"))
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_wav = os.path.join(args.out_dir, f"{stamp}-{args.lang}-seed-{args.seed}.wav")

    torch.manual_seed(args.seed)
    torch.cuda.reset_peak_memory_stats()

    from TTS.api import TTS
    import TTS as tts_pkg

    t0 = time.time()
    tts = TTS(MODEL, gpu=True)
    load_s = time.time() - t0

    t0 = time.time()
    tts.tts_to_file(text=args.text, file_path=out_wav, speaker_wav=args.ref, language=args.lang)
    gen_s = time.time() - t0

    info = sf.info(out_wav)
    ref_info = sf.info(args.ref)
    record = {
        "model": MODEL,
        "mode": "zero-shot, pretrained, no fine-tuning",
        "ref": os.path.basename(args.ref),
        "ref_seconds": round(ref_info.duration, 2),
        "text": args.text,
        "language": args.lang,
        "seed": args.seed,
        "output": os.path.basename(out_wav),
        "output_seconds": round(info.duration, 2),
        "sample_rate": info.samplerate,
        "load_seconds": round(load_s, 2),
        "generate_seconds": round(gen_s, 2),
        "peak_gpu_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "commit": git_commit(),
        "torch": torch.__version__,
        "coqui_tts": tts_pkg.__version__,
        "gpu": torch.cuda.get_device_name(0),
    }
    with open(out_wav[:-4] + ".json", "w") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
