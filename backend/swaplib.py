"""
Shared helpers for one-word replacement: audio loading, word-level
transcription, cut boundaries, splicing, plots and run records.

Used by word_swap.py (XTTS splice, runs in .venv) and neural_edit.py / app.py
(F5-TTS editing, runs in .venv-f5), so it must only depend on packages both
environments have: numpy, soundfile, torch, torchaudio, faster-whisper, matplotlib.
"""

import gc
import json
import os
import re
import subprocess
from datetime import datetime

import numpy as np
import soundfile as sf
import torch

import utils.audio_compat  # noqa: F401 — patches torchaudio.load/save
import torchaudio.functional as AF

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_ROOT = os.path.join(HERE, "outputs", "word_swap")
SR = 24000  # XTTS v2 and F5-TTS both work at 24 kHz; extract.py also normalises sources to 24 kHz
WHISPER_MODEL = "large-v3-turbo"  # ~1.6 GB; large-v3 (~3 GB) is slightly better on Arabic
FADE_S = 0.015
SNAP_S = 0.06
NEIGHBOUR_TOL_S = 0.02  # whisper boundaries are approximate; allow cuts this far into a neighbour

_DIACRITICS = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭـ]")
_PUNCT = re.compile(r"[^\w]", re.UNICODE)


def norm_ar(word: str) -> str:
    w = _DIACRITICS.sub("", word.lower())
    w = re.sub("[إأآا]", "ا", w).replace("ى", "ي").replace("ة", "ه")
    return _PUNCT.sub("", w)


def load_window(path: str, start: float, end: float) -> np.ndarray:
    """Mono 24 kHz float32 slice of a WAV/FLAC file that soundfile can read."""
    info = sf.info(path)
    a, sr = sf.read(path, start=int(start * info.samplerate), stop=int(end * info.samplerate),
                    dtype="float32", always_2d=True)
    a = a.mean(axis=1)
    if sr != SR:
        a = AF.resample(torch.from_numpy(a), sr, SR).numpy()
    return a


def load_any(path: str) -> np.ndarray:
    """Any audio/video file ffmpeg can read → mono 24 kHz float32.
    Uses the ffmpeg command line: torchcodec and PyAV cannot load this system's FFmpeg 9."""
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
        check=True, capture_output=True,
    ).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


def free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class Transcriber:
    """faster-whisper with word timestamps. Keep one instance to avoid reloading."""

    def __init__(self, device: str = "cuda", model: str = WHISPER_MODEL):
        from faster_whisper import WhisperModel

        self.device, self.model_name = device, model
        compute = "float16" if device == "cuda" else "int8"
        self.model = WhisperModel(model, device=device, compute_type=compute)

    def words(self, audio: np.ndarray, language: str | None = "ar") -> list[dict]:
        """Word-level transcript of a 24 kHz mono array. Times are relative to the array."""
        a16 = AF.resample(torch.from_numpy(audio), SR, 16000).numpy()
        segments, _ = self.model.transcribe(a16, language=language, word_timestamps=True, beam_size=5)
        return [
            {"word": w.word.strip(), "start": float(w.start), "end": float(w.end), "prob": float(w.probability)}
            for seg in segments for w in seg.words
        ]

    def close(self):
        del self.model
        free_gpu()


def transcribe(audio: np.ndarray, language: str | None = "ar", device: str = "cuda") -> list[dict]:
    """One-off transcription that frees the model afterwards."""
    t = Transcriber(device)
    try:
        return t.words(audio, language)
    finally:
        t.close()


def snap(audio: np.ndarray, t: float, lo_t: float, hi_t: float) -> float:
    """Move a boundary to the quietest 10 ms frame within ±SNAP_S, so cuts land in gaps.
    lo_t/hi_t keep it out of neighbouring words."""
    hop = int(0.01 * SR)
    lo = max(0, int(max(t - SNAP_S, lo_t) * SR))
    hi = min(len(audio), int(min(t + SNAP_S, hi_t) * SR))
    if hi - lo < 2 * hop:
        return t
    frames = [(i, float(np.sqrt(np.mean(audio[i:i + hop] ** 2)))) for i in range(lo, hi - hop, hop // 2)]
    i, _ = min(frames, key=lambda f: f[1])
    return (i + hop / 2) / SR


def word_bounds(audio: np.ndarray, words: list[dict], k: int) -> tuple[float, float]:
    """Snapped start/end of words[k], never crossing into words[k-1] or words[k+1]."""
    w = words[k]
    mid = (w["start"] + w["end"]) / 2
    prev_end = words[k - 1]["end"] - NEIGHBOUR_TOL_S if k > 0 else 0.0
    next_start = words[k + 1]["start"] + NEIGHBOUR_TOL_S if k + 1 < len(words) else len(audio) / SR
    return snap(audio, w["start"], prev_end, mid), snap(audio, w["end"], mid, next_start)


def locate_word(swords: list[dict], new_word: str, k: int, n_orig: int) -> tuple[int | None, str]:
    """Index of new_word in a transcript of synthetic speech, preferring exact matches near position k."""
    key = norm_ar(new_word)
    hits = [i for i, w in enumerate(swords) if norm_ar(w["word"]) == key]
    hits = hits or [i for i, w in enumerate(swords)
                    if key and norm_ar(w["word"]) and (key in norm_ar(w["word"]) or norm_ar(w["word"]) in key)]
    if hits:
        return min(hits, key=lambda i: abs(i - k)), "text match"
    if len(swords) == n_orig:
        return k, "position (no text match)"
    return None, "not found"


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


def plot(orig, fake, a, b_orig, b_fake, path, pad=1.2, fake_title="fake: one word replaced"):
    """Spectrograms of original and fake around the replaced word, on one colour scale."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import mlab

    lo = max(0.0, a - pad)
    spec = dict(NFFT=1024, Fs=SR, noverlap=896)
    p_db = 10 * np.log10(mlab.specgram(orig, **spec)[0] + 1e-12)
    vmin, vmax = np.percentile(p_db, [5, 99.5])

    fig, axes = plt.subplots(2, 1, figsize=(10, 4.6), sharex=True, sharey=True)
    for ax, x, b, title in [(axes[0], orig, b_orig, "original"), (axes[1], fake, b_fake, fake_title)]:
        hi = min(len(x) / SR, b + pad)
        seg = x[int(lo * SR):int(hi * SR)]
        ax.specgram(seg, xextent=(lo, hi), cmap="magma", vmin=vmin, vmax=vmax, **spec)
        for t in (a, b):
            ax.axvline(t, color="white", lw=0.9, ls=(0, (3, 2)))
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_ylabel("Hz")
        ax.set_ylim(0, 8000)
    axes[1].set_xlabel("seconds into the clip")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def git_commit() -> str:
    try:
        dirty = subprocess.run(["git", "-C", HERE, "diff", "--quiet", "HEAD", "--", "."]).returncode != 0
        head = subprocess.check_output(["git", "-C", HERE, "rev-parse", "--short", "HEAD"], text=True).strip()
        return head + ("+uncommitted" if dirty else "")
    except Exception:
        return "unknown"


def new_run_dir(tag: str = "") -> str:
    run = datetime.now().strftime("%Y%m%d-%H%M%S") + (f"-{tag}" if tag else "")
    out = os.path.join(OUT_ROOT, run)
    os.makedirs(out, exist_ok=True)
    return out


def write_record(out: str, record: dict):
    with open(os.path.join(out, "record.json"), "w") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
