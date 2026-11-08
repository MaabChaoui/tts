"""
Word-level voice deepfake by neural speech editing (F5-TTS infilling).

The span of the old word is blanked in the real recording; the model is given
the whole edited sentence and fills the gap so it fits the audio around it.
Default weights: SILMA TTS v1 (Arabic/English, F5-TTS v1.1.7 architecture,
Apache-2.0). Runs in .venv-f5. Follows f5_tts/infer/speech_edit.py.

Two outputs per edit, mirroring the image pilot:
  regen  the whole clip as decoded by the vocoder (every sample is resynthesised)
  paste  the original clip with only the regenerated span spliced in

Test on a window of a long file:
    .venv-f5/bin/python neural_edit.py --wav data/audio/NatGeoHistory/lE2B6DdFZBI.wav \
        --start 201.6 --end 213.3 --word-index 6 --new-word "الألمان"
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

SILMA_REPO = "silma-ai/silma-tts"
NFE_STEPS = 32
CFG_STRENGTH = 2.0
SWAY_COEF = -1.0
TARGET_RMS = 0.1
HOP = 256


class F5Editor:
    def __init__(self, repo: str = SILMA_REPO, device: str = "cuda"):
        from huggingface_hub import hf_hub_download
        from omegaconf import OmegaConf
        from f5_tts.infer.utils_infer import load_checkpoint, load_vocoder
        from f5_tts.model import CFM, DiT
        from f5_tts.model.utils import get_tokenizer

        self.repo, self.device = repo, device
        cfg = OmegaConf.load(hf_hub_download(repo, "config.yaml"))
        vocab_char_map, vocab_size = get_tokenizer(hf_hub_download(repo, "vocab.txt"), "custom")
        mel = cfg.model.mel_spec
        self.model = CFM(
            transformer=DiT(**cfg.model.arch, text_num_embeds=vocab_size, mel_dim=mel.n_mel_channels),
            mel_spec_kwargs=dict(n_fft=mel.n_fft, hop_length=mel.hop_length, win_length=mel.win_length,
                                 n_mel_channels=mel.n_mel_channels, target_sample_rate=mel.target_sample_rate,
                                 mel_spec_type=mel.mel_spec_type),
            odeint_kwargs=dict(method="euler"),
            vocab_char_map=vocab_char_map,
        ).to(device)
        self.model = load_checkpoint(self.model, hf_hub_download(repo, "model.pt"), device, use_ema=True)
        self.vocoder = load_vocoder("vocos", is_local=False, device=device)
        assert mel.target_sample_rate == sl.SR and mel.hop_length == HOP

    @torch.inference_mode()
    def edit(self, orig: np.ndarray, a: float, b: float, new_dur: float, text: str, seed: int = 0) -> np.ndarray:
        """Blank orig[a:b] (seconds), make room for new_dur seconds, and regenerate with `text`
        (the full edited sentence). Returns the whole resynthesised clip."""
        from f5_tts.model.utils import convert_char_to_pinyin

        audio = torch.from_numpy(orig).float()[None]
        clip_rms = float(torch.sqrt(torch.mean(audio ** 2)))
        if clip_rms < TARGET_RMS:
            audio = audio * TARGET_RMS / clip_rms
        ia, ib, n_new = round(a * sl.SR), round(b * sl.SR), round(new_dur * sl.SR)
        cond = torch.cat([audio[:, :ia], torch.zeros(1, n_new), audio[:, ib:]], dim=-1)
        keep = torch.cat([torch.ones(1, round(ia / HOP), dtype=torch.bool),
                          torch.zeros(1, round(n_new / HOP), dtype=torch.bool)], dim=-1)
        frames = cond.shape[-1] // HOP
        keep = torch.nn.functional.pad(keep, (0, frames - keep.shape[-1] + 1), value=True)

        mel, _ = self.model.sample(
            cond=cond.to(self.device), text=convert_char_to_pinyin([text]), duration=frames,
            steps=NFE_STEPS, cfg_strength=CFG_STRENGTH, sway_sampling_coef=SWAY_COEF, seed=seed,
            edit_mask=keep.to(self.device),
        )
        wave = self.vocoder.decode(mel.to(torch.float32).permute(0, 2, 1)).cpu()[0].numpy()
        if clip_rms < TARGET_RMS:
            wave = wave * clip_rms / TARGET_RMS
        return wave.astype(np.float32)


def new_word_duration(old_dur: float, old_word: str, new_word: str) -> float:
    """Scale the old word's duration by the ratio of letters, within limits."""
    ratio = len(sl.norm_ar(new_word) or new_word) / max(1, len(sl.norm_ar(old_word) or old_word))
    return old_dur * float(np.clip(ratio, 0.75, 1.4))


SPAN_MARGIN_S = 0.03


def edit_span(orig, words, k, context: int = 0) -> tuple[float, float]:
    """Span to regenerate: the snapped word bounds, widened to cover whisper's full word plus a
    small margin, but never past the neighbours. Leaving part of the old word (e.g. its first
    vowel) in place makes the model continue from it and say a different word."""
    k0, k1 = max(0, k - context), min(len(words) - 1, k + context)
    a, _ = sl.word_bounds(orig, words, k0)
    _, b = sl.word_bounds(orig, words, k1)
    lo = words[k0 - 1]["end"] - sl.NEIGHBOUR_TOL_S if k0 > 0 else 0.0
    hi = words[k1 + 1]["start"] + sl.NEIGHBOUR_TOL_S if k1 + 1 < len(words) else len(orig) / sl.SR
    return (max(lo, min(a, words[k0]["start"]) - SPAN_MARGIN_S),
            min(hi, max(b, words[k1]["end"]) + SPAN_MARGIN_S))


def edit_word(editor: F5Editor, orig, words, k, new_word, out, text=None, seed=0, transcriber=None,
              language="ar", new_dur=None, context: int = 0) -> dict:
    """Neural edit on a clip with known words. Writes WAVs, plots and record.json to out."""
    t_run = time.time()
    new_words = [w["word"] for w in words]
    new_words[k] = new_word
    orig_text = " ".join(w["word"] for w in words)
    new_text = text or " ".join(new_words)

    a, b = edit_span(orig, words, k, context)
    wa, wb = edit_span(orig, words, k)
    word_dur = new_dur or new_word_duration(wb - wa, words[k]["word"], new_word)
    new_dur = (b - a) - (wb - wa) + word_dur  # neighbours keep their length; only the word changes
    regen = editor.edit(orig, a, b, new_dur, new_text, seed)

    # The regenerated span sits at [a, a + new_dur] in the regenerated clip; paste only that.
    n_piece = len(regen) - (len(orig) - (round(b * sl.SR) - round(a * sl.SR)))
    piece = regen[round(a * sl.SR):round(a * sl.SR) + n_piece]
    paste = sl.splice(orig, a, b, piece)

    sf.write(os.path.join(out, "original.wav"), orig, sl.SR)
    sf.write(os.path.join(out, "regen.wav"), regen, sl.SR)
    sf.write(os.path.join(out, "fake.wav"), paste, sl.SR)
    b_new = a + len(piece) / sl.SR
    sl.plot(orig, paste, a, b, b_new, os.path.join(out, "splice.png"), fake_title="fake: new word pasted into the original")
    sl.plot(orig, regen, a, b, b_new, os.path.join(out, "regen.png"), fake_title="fake: whole clip regenerated")

    heard = None
    if transcriber is not None:
        heard = {name: " ".join(w["word"] for w in transcriber.words(x, language))
                 for name, x in [("fake", paste), ("regen", regen)]}

    import f5_tts
    import importlib.metadata as md
    record = {
        "method": "neural edit (F5-TTS infilling)",
        "weights": editor.repo,
        "language": language,
        "original_text": orig_text,
        "fake_text": new_text,
        "word_index": k,
        "original_word": words[k]["word"],
        "new_word": new_word,
        "original_word_s_whisper": [words[k]["start"], words[k]["end"]],
        "edited_span_s": [round(a, 3), round(b, 3)],
        "new_word_s": round(new_dur, 3),
        "whisper_hears": heard,
        "nfe_steps": NFE_STEPS,
        "cfg_strength": CFG_STRENGTH,
        "sway_sampling_coef": SWAY_COEF,
        "span_margin_s": SPAN_MARGIN_S,
        "context_words": context,
        "crossfade_s": sl.FADE_S,
        "seed": seed,
        "f5_tts": md.version("f5-tts"),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "commit": sl.git_commit(),
        "seconds": round(time.time() - t_run, 1),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    sl.write_record(out, record)
    return record


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wav", required=True)
    p.add_argument("--start", type=float, required=True)
    p.add_argument("--end", type=float, required=True)
    p.add_argument("--word-index", type=int, required=True)
    p.add_argument("--new-word", required=True)
    p.add_argument("--text", help="full edited sentence; default: transcript with the word replaced")
    p.add_argument("--lang", default="ar")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--new-dur", type=float, help="seconds for the new word; default: scaled from the old word")
    p.add_argument("--context-words", type=int, default=0, help="also regenerate this many words on each side")
    args = p.parse_args()

    orig = sl.load_window(args.wav, args.start, args.end)
    tr = sl.Transcriber("cuda")
    words = tr.words(orig, args.lang)
    for i, w in enumerate(words):
        print(f"{i:3d} {w['start']:6.2f} {w['end']:6.2f} {w['word']}")
    editor = F5Editor()
    out = sl.new_run_dir("f5")
    record = edit_word(editor, orig, words, args.word_index, args.new_word, out, args.text, args.seed, tr, args.lang,
                       args.new_dur, args.context_words)
    record = {"source": os.path.relpath(os.path.abspath(args.wav), sl.HERE), "window_s": [args.start, args.end], **record}
    sl.write_record(out, record)
    print(json.dumps(record, ensure_ascii=False, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
