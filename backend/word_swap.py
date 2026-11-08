"""
Word-level voice deepfake (partial spoof) by splicing: the baseline method.

Replaces one word in a real recorded sentence with the same speaker's cloned
voice saying a different word. XTTS v2 clones the voice zero-shot and says the
edited sentence; the new word is cut from that and spliced into the real audio.
Runs in .venv (coqui-tts). Shared helpers are in swaplib.py.

Commands:
    words   list words with timings in a window of a long file
              .venv/bin/python word_swap.py words --wav data/audio/dahih/X.wav --start 60 --end 180
    swap    replace one word in a window of a long file
              .venv/bin/python word_swap.py swap --wav data/audio/dahih/X.wav \
                  --start 75.2 --end 81.9 --word-index 4 --new-word "مليار"
    edit    replace one word in a whole clip whose words are already known (used by app.py)
    replot  redraw splice.png for an existing run

Outputs go to outputs/word_swap/<run>/ (gitignored). Real people's voices:
keep everything local, research use only.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, os.path.dirname(__file__))
import swaplib as sl

os.environ["COQUI_TOS_AGREED"] = "1"

XTTS_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"


def xtts_say(text: str, ref_path: str, language: str) -> np.ndarray:
    from TTS.api import TTS

    tts = TTS(XTTS_MODEL, gpu=torch.cuda.is_available())
    try:
        return np.asarray(tts.tts(text=text, speaker_wav=ref_path, language=language), dtype=np.float32)
    finally:
        del tts
        sl.free_gpu()


def splice_word(orig, words, k, new_word, out, language="ar", text=None, ref=None, seed=0,
                whisper_device="cuda") -> dict:
    """XTTS splice on a clip with known words. Writes WAVs, splice.png and record.json to out."""
    t_run = time.time()
    torch.manual_seed(seed)
    new_words = [w["word"] for w in words]
    new_words[k] = new_word
    orig_text = " ".join(w["word"] for w in words)
    new_text = text or " ".join(new_words)

    ref_path = os.path.join(out, "reference.wav")
    sf.write(ref_path, orig if ref is None else ref, sl.SR)
    synth = xtts_say(new_text, ref_path, language)
    sf.write(os.path.join(out, "synth_full.wav"), synth, sl.SR)

    swords = sl.transcribe(synth, language, whisper_device)
    j, located_by = sl.locate_word(swords, new_word, k, len(words))
    if j is None:
        raise RuntimeError(f"Could not find '{new_word}' in the synthetic sentence: {' '.join(w['word'] for w in swords)}")

    oa, ob = sl.word_bounds(orig, words, k)
    sa, sb = sl.word_bounds(synth, swords, j)
    piece = synth[int(sa * sl.SR):int(sb * sl.SR)]
    gain = sl.rms(orig[int(oa * sl.SR):int(ob * sl.SR)]) / sl.rms(piece)
    piece = piece * gain
    fake = sl.splice(orig, oa, ob, piece)

    sf.write(os.path.join(out, "original.wav"), orig, sl.SR)
    sf.write(os.path.join(out, "fake.wav"), fake, sl.SR)
    sl.plot(orig, fake, oa, ob, oa + len(piece) / sl.SR, os.path.join(out, "splice.png"))

    import TTS as tts_pkg
    record = {
        "method": "splice (XTTS v2)",
        "language": language,
        "original_text": orig_text,
        "fake_text": new_text,
        "word_index": k,
        "original_word": words[k]["word"],
        "new_word": new_word,
        "original_word_s_whisper": [words[k]["start"], words[k]["end"]],
        "original_word_s_snapped": [round(oa, 3), round(ob, 3)],
        "synth_word": swords[j]["word"],
        "synth_word_s_snapped": [round(sa, 3), round(sb, 3)],
        "synth_word_located_by": located_by,
        "synth_transcript": " ".join(w["word"] for w in swords),
        "gain": round(gain, 3),
        "crossfade_s": sl.FADE_S,
        "snap_s": sl.SNAP_S,
        "neighbour_tol_s": sl.NEIGHBOUR_TOL_S,
        "seed": seed,
        "xtts": XTTS_MODEL,
        "whisper": sl.WHISPER_MODEL,
        "whisper_device": whisper_device,
        "coqui_tts": tts_pkg.__version__,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "commit": sl.git_commit(),
        "seconds": round(time.time() - t_run, 1),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    sl.write_record(out, record)
    return record


def cmd_words(args):
    audio = sl.load_window(args.wav, args.start, args.end)
    for i, w in enumerate(sl.transcribe(audio, args.lang, args.whisper_device)):
        print(f"{i:4d}  {args.start + w['start']:8.2f}  {args.start + w['end']:8.2f}  {w['prob']:.2f}  {w['word']}")


def cmd_swap(args):
    orig = sl.load_window(args.wav, args.start, args.end)
    words = sl.transcribe(orig, args.lang, args.whisper_device)
    if not 0 <= args.word_index < len(words):
        sys.exit(f"--word-index must be in [0, {len(words) - 1}]; the sentence has {len(words)} words")
    ref = sl.load_window(args.wav, args.ref_start, args.ref_end) if args.ref_start is not None else None
    out = sl.new_run_dir()
    record = splice_word(orig, words, args.word_index, args.new_word, out, args.lang, ref=ref,
                         seed=args.seed, whisper_device=args.whisper_device)
    record = {"source": os.path.relpath(os.path.abspath(args.wav), sl.HERE), "window_s": [args.start, args.end],
              "reference_window_s": [args.ref_start, args.ref_end] if ref is not None else "same as window", **record}
    sl.write_record(out, record)
    print(json.dumps(record, ensure_ascii=False, indent=2))
    print(f"\nWrote {out}")


def cmd_edit(args):
    """For app.py: the clip is already converted and transcribed."""
    orig, _ = sf.read(args.clip, dtype="float32")
    with open(args.words_json) as f:
        words = json.load(f)
    try:
        record = splice_word(orig, words, args.word_index, args.new_word, args.out, args.lang,
                             text=args.text, seed=args.seed, whisper_device=args.whisper_device)
        print("RESULT " + json.dumps({"ok": True, "record": record}, ensure_ascii=False))
    except Exception as e:
        print("RESULT " + json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"}, ensure_ascii=False))


def cmd_replot(args):
    with open(os.path.join(args.run, "record.json")) as f:
        r = json.load(f)
    orig, _ = sf.read(os.path.join(args.run, "original.wav"), dtype="float32")
    fake, _ = sf.read(os.path.join(args.run, "fake.wav"), dtype="float32")
    a, b = r["original_word_s_snapped"]
    sa, sb = r["synth_word_s_snapped"]
    sl.plot(orig, fake, a, b, a + (sb - sa), os.path.join(args.run, "splice.png"))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("words", help="list words with timings in a window")
    s = sub.add_parser("swap", help="replace one word in a window of a long file")
    e = sub.add_parser("edit", help="replace one word in a transcribed clip (used by app.py)")
    r = sub.add_parser("replot", help="redraw splice.png for an existing run")
    r.add_argument("--run", required=True)
    for q in (w, s):
        q.add_argument("--wav", required=True)
        q.add_argument("--start", type=float, required=True)
        q.add_argument("--end", type=float, required=True)
    for q in (s, e):
        q.add_argument("--word-index", type=int, required=True)
        q.add_argument("--new-word", required=True)
        q.add_argument("--seed", type=int, default=0)
    s.add_argument("--ref-start", type=float)
    s.add_argument("--ref-end", type=float)
    e.add_argument("--clip", required=True, help="24 kHz mono WAV")
    e.add_argument("--words-json", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--text", help="full edited text for XTTS; default: transcript with the word replaced")
    for q in (w, s, e):
        q.add_argument("--lang", default="ar")
        q.add_argument("--whisper-device", choices=["cuda", "cpu"], default="cuda")
    args = p.parse_args()
    {"words": cmd_words, "swap": cmd_swap, "edit": cmd_edit, "replot": cmd_replot}[args.cmd](args)


if __name__ == "__main__":
    main()
