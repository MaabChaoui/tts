"""
Word-level voice deepfake (partial spoof) demo.

Replaces one word in a real recorded sentence with the same speaker's cloned
voice saying a different word. Uses pretrained models only: faster-whisper for
word timings, XTTS v2 zero-shot cloning for the new word.

Steps:
    1. List words with timings in a window of a source file:
         .venv/bin/python word_swap.py words --wav data/audio/dahih/X.wav --start 60 --end 180
    2. Pick one sentence (start/end) and the index of the word to replace:
         .venv/bin/python word_swap.py swap --wav data/audio/dahih/X.wav \
             --start 75.2 --end 81.9 --word-index 4 --new-word "مليار"

Outputs go to outputs/word_swap/<run>/ (gitignored): original.wav, fake.wav,
synth_full.wav, splice.png and record.json. Real people's voices: keep
everything local, research use only.
"""

import argparse
import gc
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, os.path.dirname(__file__))
import utils.audio_compat  # noqa: F401 — patches torchaudio.load/save
import torchaudio.functional as AF

os.environ["COQUI_TOS_AGREED"] = "1"

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_ROOT = os.path.join(HERE, "outputs", "word_swap")
SR = 24000  # XTTS v2 output rate; extract.py also normalises sources to 24 kHz
XTTS_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"
WHISPER_MODEL = "large-v3-turbo"  # ~1.6 GB; large-v3 (~3 GB) is slightly better on Arabic
WHISPER_DEVICE = "cuda"  # set by --whisper-device; "cpu" leaves the GPU to other jobs
FADE_S = 0.015
SNAP_S = 0.06

_DIACRITICS = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭـ]")
_PUNCT = re.compile(r"[^\w]", re.UNICODE)


def norm_ar(word: str) -> str:
    w = _DIACRITICS.sub("", word)
    w = re.sub("[إأآا]", "ا", w).replace("ى", "ي").replace("ة", "ه")
    return _PUNCT.sub("", w)


def load_window(path: str, start: float, end: float) -> np.ndarray:
    info = sf.info(path)
    a, sr = sf.read(path, start=int(start * info.samplerate), stop=int(end * info.samplerate),
                    dtype="float32", always_2d=True)
    a = a.mean(axis=1)
    if sr != SR:
        a = AF.resample(torch.from_numpy(a), sr, SR).numpy()
    return a


def free_gpu():
    gc.collect()
    torch.cuda.empty_cache()


def transcribe(audio: np.ndarray) -> list[dict]:
    """Word-level transcript of a 24 kHz mono array. Times are relative to the array."""
    from faster_whisper import WhisperModel

    compute = "float16" if WHISPER_DEVICE == "cuda" else "int8"
    model = WhisperModel(WHISPER_MODEL, device=WHISPER_DEVICE, compute_type=compute)
    a16 = AF.resample(torch.from_numpy(audio), SR, 16000).numpy()
    segments, _ = model.transcribe(a16, language="ar", word_timestamps=True, beam_size=5)
    words = [
        {"word": w.word.strip(), "start": float(w.start), "end": float(w.end), "prob": float(w.probability)}
        for seg in segments for w in seg.words
    ]
    del model
    free_gpu()
    return words


def snap(audio: np.ndarray, t: float) -> float:
    """Move a boundary to the quietest 10 ms frame within ±SNAP_S, so cuts land in gaps."""
    hop = int(0.01 * SR)
    lo, hi = max(0, int((t - SNAP_S) * SR)), min(len(audio), int((t + SNAP_S) * SR))
    if hi - lo < 2 * hop:
        return t
    frames = [(i, float(np.sqrt(np.mean(audio[i:i + hop] ** 2)))) for i in range(lo, hi - hop, hop // 2)]
    i, _ = min(frames, key=lambda f: f[1])
    return (i + hop / 2) / SR


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2)) + 1e-9)


def splice(orig: np.ndarray, a: float, b: float, piece: np.ndarray) -> np.ndarray:
    """Replace orig[a:b] (seconds) with piece, with equal-power crossfades at both joins."""
    f = int(FADE_S * SR)
    ia, ib = int(a * SR), int(b * SR)
    t = np.linspace(0, np.pi / 2, f, dtype=np.float32)
    fade_in, fade_out = np.sin(t), np.cos(t)
    head, tail = orig[:ia + f].copy(), orig[ib - f:].copy()
    piece = piece.copy()
    head[-f:] = head[-f:] * fade_out + piece[:f] * fade_in
    tail[:f] = piece[-f:] * fade_out + tail[:f] * fade_in
    return np.concatenate([head, piece[f:-f], tail])


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "-C", HERE, "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def cmd_words(args):
    audio = load_window(args.wav, args.start, args.end)
    words = transcribe(audio)
    for i, w in enumerate(words):
        print(f"{i:4d}  {args.start + w['start']:8.2f}  {args.start + w['end']:8.2f}  {w['prob']:.2f}  {w['word']}")


def cmd_swap(args):
    t_run = time.time()
    torch.manual_seed(args.seed)
    run = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = os.path.join(OUT_ROOT, run)
    os.makedirs(out, exist_ok=True)

    orig = load_window(args.wav, args.start, args.end)
    words = transcribe(orig)
    if not 0 <= args.word_index < len(words):
        sys.exit(f"--word-index must be in [0, {len(words) - 1}]; the sentence has {len(words)} words")
    target = words[args.word_index]
    new_words = [w["word"] for w in words]
    new_words[args.word_index] = args.new_word
    orig_text, new_text = " ".join(w["word"] for w in words), " ".join(new_words)

    # Reference for cloning: a separate window if given, else the sentence itself.
    ref = load_window(args.wav, args.ref_start, args.ref_end) if args.ref_start is not None else orig
    ref_path = os.path.join(out, "reference.wav")
    sf.write(ref_path, ref, SR)

    from TTS.api import TTS
    import TTS as tts_pkg

    tts = TTS(XTTS_MODEL, gpu=True)
    synth = np.asarray(tts.tts(text=new_text, speaker_wav=ref_path, language="ar"), dtype=np.float32)
    del tts
    free_gpu()

    # Locate the new word in the synthetic sentence.
    swords = transcribe(synth)
    key = norm_ar(args.new_word)
    hits = [i for i, w in enumerate(swords) if norm_ar(w["word"]) == key]
    hits = hits or [i for i, w in enumerate(swords) if key and norm_ar(w["word"]) and (key in norm_ar(w["word"]) or norm_ar(w["word"]) in key)]
    if hits:
        j = min(hits, key=lambda i: abs(i - args.word_index))
        located_by = "text match"
    elif len(swords) == len(words):
        j = args.word_index
        located_by = "position (no text match)"
    else:
        sf.write(os.path.join(out, "synth_full.wav"), synth, SR)
        sys.exit(f"Could not find '{args.new_word}' in synthetic transcript: {[w['word'] for w in swords]}")

    oa, ob = snap(orig, target["start"]), snap(orig, target["end"])
    sa, sb = snap(synth, swords[j]["start"]), snap(synth, swords[j]["end"])
    piece = synth[int(sa * SR):int(sb * SR)]
    gain = rms(orig[int(oa * SR):int(ob * SR)]) / rms(piece)
    piece = piece * gain
    fake = splice(orig, oa, ob, piece)

    sf.write(os.path.join(out, "original.wav"), orig, SR)
    sf.write(os.path.join(out, "fake.wav"), fake, SR)
    sf.write(os.path.join(out, "synth_full.wav"), synth, SR)
    plot(orig, fake, oa, ob, oa + len(piece) / SR, os.path.join(out, "splice.png"))

    record = {
        "source": os.path.relpath(os.path.abspath(args.wav), HERE),
        "window_s": [args.start, args.end],
        "reference_window_s": [args.ref_start, args.ref_end] if args.ref_start is not None else "same as window",
        "original_text": orig_text,
        "fake_text": new_text,
        "word_index": args.word_index,
        "original_word": target["word"],
        "new_word": args.new_word,
        "original_word_s_whisper": [target["start"], target["end"]],
        "original_word_s_snapped": [round(oa, 3), round(ob, 3)],
        "synth_word": swords[j]["word"],
        "synth_word_s_snapped": [round(sa, 3), round(sb, 3)],
        "synth_word_located_by": located_by,
        "synth_transcript": " ".join(w["word"] for w in swords),
        "gain": round(gain, 3),
        "crossfade_s": FADE_S,
        "seed": args.seed,
        "xtts": XTTS_MODEL,
        "whisper": WHISPER_MODEL,
        "whisper_device": WHISPER_DEVICE,
        "coqui_tts": tts_pkg.__version__,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "commit": git_commit(),
        "seconds": round(time.time() - t_run, 1),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(os.path.join(out, "record.json"), "w") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    print(json.dumps(record, ensure_ascii=False, indent=2))
    print(f"\nWrote {out}")


def plot(orig, fake, a, b_orig, b_fake, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(11, 5), sharey=True)
    for ax, x, b, title in [(axes[0], orig, b_orig, "original"), (axes[1], fake, b_fake, "fake (one word replaced)")]:
        ax.specgram(x, NFFT=512, Fs=SR, noverlap=384, cmap="magma")
        ax.axvspan(a, b, color="white", alpha=0.18, lw=0)
        ax.axvline(a, color="white", lw=0.8)
        ax.axvline(b, color="white", lw=0.8)
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_ylabel("Hz")
        ax.set_ylim(0, 8000)
    axes[1].set_xlabel("seconds")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("words", help="list words with timings in a window")
    s = sub.add_parser("swap", help="replace one word in a sentence")
    for q in (w, s):
        q.add_argument("--wav", required=True)
        q.add_argument("--start", type=float, required=True)
        q.add_argument("--end", type=float, required=True)
    s.add_argument("--word-index", type=int, required=True)
    s.add_argument("--new-word", required=True)
    s.add_argument("--ref-start", type=float)
    s.add_argument("--ref-end", type=float)
    s.add_argument("--seed", type=int, default=0)
    for q in (w, s):
        q.add_argument("--whisper-device", choices=["cuda", "cpu"], default="cuda")
    args = p.parse_args()
    global WHISPER_DEVICE
    WHISPER_DEVICE = args.whisper_device
    {"words": cmd_words, "swap": cmd_swap}[args.cmd](args)


if __name__ == "__main__":
    main()
