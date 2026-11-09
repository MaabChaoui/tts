"""
Slide figure for a neural edit run: what F5-TTS is given and what it returns.

Three spectrograms on one colour scale, around the edited span:
  original | the model's input (span blanked, room made for the new word) | regenerated clip

    .venv-f5/bin/python infill_figure.py outputs/word_swap/20261109-003801-f5
Writes infill.png into the run directory (gitignored with the rest of outputs/).
"""

import json
import os
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(__file__))
import swaplib as sl


def main(run: str, pad: float = 0.8):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import mlab

    rec = json.load(open(os.path.join(run, "record.json")))
    orig, _ = sf.read(os.path.join(run, "original.wav"), dtype="float32")
    regen, _ = sf.read(os.path.join(run, "regen.wav"), dtype="float32")
    a, b = rec["edited_span_s"]
    new_dur = rec["new_word_s"]
    # Same conditioning as F5Editor.edit: the span is replaced by silence of the new length.
    cond = np.concatenate([orig[:round(a * sl.SR)], np.zeros(round(new_dur * sl.SR), np.float32), orig[round(b * sl.SR):]])

    spec = dict(NFFT=1024, Fs=sl.SR, noverlap=896)
    p_db = 10 * np.log10(mlab.specgram(orig, **spec)[0] + 1e-12)
    vmin, vmax = np.percentile(p_db, [5, 99.5])
    lo = max(0.0, a - pad)
    panels = [(orig, b, "original"), (cond, a + new_dur, "given to the model: word blanked"),
              (regen, a + new_dur, "returned: span filled in context")]

    fig, axes = plt.subplots(1, 3, figsize=(12, 2.9), sharey=True)
    for ax, (x, end, title) in zip(axes, panels):
        hi = min(len(x) / sl.SR, end + pad)
        seg = x[int(lo * sl.SR):int(hi * sl.SR)]
        seg = seg + 1e-6 * np.random.default_rng(0).standard_normal(len(seg)).astype(np.float32)  # log of exact zeros
        ax.specgram(seg, xextent=(lo, hi), cmap="magma", vmin=vmin, vmax=vmax, **spec)
        for t in (a, end):
            ax.axvline(t, color="white", lw=0.9, ls=(0, (3, 2)))
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_ylim(0, 8000)
        ax.set_xlabel("s")
    axes[0].set_ylabel("Hz")
    fig.tight_layout()
    path = os.path.join(run, "infill.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    main(sys.argv[1])
