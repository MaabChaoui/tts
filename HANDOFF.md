# /resHANDOFF.md — TTS Project Session Transfer

## Latest: 2026-11-09, neural editing and the one-word tool

**Done**
- Second environment `backend/.venv-f5` (F5-TTS 1.1.7, gradio 5.35, faster-whisper; exact versions in `backend/requirements-f5.lock`). Separate from `.venv` because f5-tts pins numpy 1.26.4. Needed `pyarrow<22`. Hugging Face's Xet download stalled; `HF_HUB_DISABLE_XET=1` fixed it.
- Weights: SILMA TTS v1 (`silma-ai/silma-tts`, Arabic/English, 163M parameters, F5-TTS architecture, Apache-2.0) and the Vocos vocoder.
- `backend/neural_edit.py`: F5-TTS infilling. The old word's span is blanked and regenerated from the full edited sentence. Outputs `fake.wav` (only the new span pasted into the original) and `regen.wav` (whole clip through the vocoder).
- `backend/swaplib.py`: shared helpers. `word_swap.py` now uses it; regression run 20261109-002018 matches run 20261108-232616 exactly.
- `backend/app.py` + `run_app.sh`: local Gradio tool at http://127.0.0.1:7861. Upload or record ≤ 30 s, transcribe, click a word, type the new one, generate with neural edit or XTTS splice. About 11 s per neural edit, 40 s per splice (XTTS reloads in a helper process). Holds about 2.8 GB of GPU memory.

**Editing test (Whisper and spectrograms only; nobody has listened yet)**
- NatGeo, الحلفاء → الألمان: works with 0 or 1 context words (runs 20261109-002717-f5, -003235-f5). Spectrogram of the edit is more continuous with its neighbours than the XTTS splice.
- الدحيح, أمريكا → فرنسا: fails with 0 context words whatever the duration ("في أن", "أنسان", "فيه"). Works once the neighbouring word on each side is regenerated too (context 1 or 2, runs 20261109-003125-f5 to -003208-f5), and the clip needs words after the edit. Cost: neighbours can drift ("وهو النظام" → "والنظام", "يواجه" → "واجه").
- Numbers: Whisper sometimes writes digits; type them as words in the sentence box.

**Not done**
- No listening test; no measurement of the tells; no detector run on these fakes.

---

## 2026-11-08 (new machine, RTX 3060 Laptop 6 GB)

**Done**
- Rebuilt `backend/.venv` (Python 3.11). Exact versions in `backend/requirements.lock`. Fixes: transformers `<5` (5.x breaks coqui-tts), `six` added, torchcodec cu128 build installed only to pass coqui-tts's import check; it cannot load here (system FFmpeg 9), so `utils/audio_compat.py` still does audio IO. faster-whisper also fails on file paths for the same reason, so pass numpy arrays.
- Re-downloaded one video per speaker (audio only; provenance in `backend/sources.txt`) and ran `pipeline.extract`. Now 1.22 h, not the old 15.8 h.
- Models cached: XTTS v2 (`~/.local/share/tts`), faster-whisper large-v3-turbo (`~/.cache/huggingface`).
- `backend/word_swap.py`: replaces one word in a real sentence with the zero-shot cloned voice (partial spoof). Two working examples: NatGeo الحلفاء→الألمان (run 20261108-232616) and dahih أمريكا→فرنسا (run 20261108-232639). Whisper reads both fakes as the edited sentence. Listening page: `backend/outputs/word_swap/index.html`.

**Not done**
- No pipeline stage after extraction has run (transcribe, diarize, segment, clip, dataset, train). Fine-tuning not started.
- pyannote not installed in the new venv (HF token is in `.env`).
- Nobody has listened to the fakes yet; quality judged only by Whisper and spectrograms.

**Note**
- The GitHub remote is public by design (Maab, 2026-11-08). Audio, transcripts and generated clips stay in gitignored `data/` and `outputs/`.

---

## Earlier sessions

## Project Overview

**Goal:** Multi-style Arabic documentary narration engine — fine-tune XTTS v2 on NatGeo/dahih narration audio to clone voices and styles.

**Stack:** Python 3.11 · PyTorch 2.11+cu128 · coqui-tts 0.27.5 · pyannote.audio 4.0.4 · openai-whisper · faster-whisper · FastAPI · Next.js (later)

**Hardware:** RTX 3050 6GB, 16GB RAM, Arch Linux, kernel 6.19.x

**Key directories:**

```
backend/
  pipeline/     # data processing scripts (extract.py done, rest TODO)
  training/     # XTTS fine-tuning (TODO)
  api/          # FastAPI server (TODO)
  utils/        # audio_compat.py — critical torchaudio patch
  data/
    raw/{speaker}/*.mp3   # source files (gitignored)
    audio/{speaker}/*.wav # extracted (gitignored)
    transcripts/          # whisper output (TODO)
    segments/             # speaker segments (TODO)
    clips/                # 3-8s training clips (TODO)
  .env          # HF_TOKEN (gitignored)
  .venv/        # Python venv (gitignored)
PLAN.md         # detailed phase-by-phase technical plan
GENERAL_PLAN.md # high-level project goals
```

---

## Files Modified This Session

| File                              | Change                                                                   |
| --------------------------------- | ------------------------------------------------------------------------ |
| `backend/smoke_test.py`         | Created: tests Whisper/pyannote/XTTS v2 sequentially with VRAM reporting |
| `backend/utils/audio_compat.py` | Created: monkey-patches torchaudio.load/save to use soundfile            |
| `backend/pipeline/extract.py`   | Created: MP3→WAV extraction (mono, 24kHz, 16-bit PCM)                   |
| `backend/validate_gpu.py`       | Verified (created by prior session)                                      |
| `backend/requirements.txt`      | Verified (created by prior session)                                      |
| `.gitignore`                    | Created: covers .venv, .env, data/, models/, outputs/                    |
| `memory/`                       | Created: persistent memory files (user profile, known issues)            |

---

## Current Status

**Completed:**

- ✅ Phase 0.1 — Directory structure, venv, GPU validated (RTX 3050 6GB confirmed)
- ✅ Phase 0.2 — HF token set, all 3 pyannote gated repos accepted
- ✅ Phase 0.3 — Smoke tests passing: Whisper PASS, pyannote PASS, XTTS v2 PASS
- ✅ Phase 0.4 — Git init, `.gitignore`, initial commit on `main`
- ✅ Phase 2.1 — Audio extraction: 23 files, 15.8h total → `data/audio/` (2.6 GB)

**Data collected:**

- `NatGeoHistory`: 11 files, 10.2h (WW2 docs, crime docs, history)
- `dahih`: 12 files, 5.6h (Egyptian educational/history YouTube channel "الدحيح")

**Blockers encountered (resolved):**

- `torchcodec 0.11` needs `libnvrtc.so.13` (system has CUDA 12.8 = `.so.12`) → patched via `utils/audio_compat.py`
- pyannote 4.x API: use `token=` not `use_auth_token=`; pass `{"waveform": tensor, "sample_rate": int}` dict instead of file path
- Disk full during first extraction run → user freed 17GB, re-ran successfully

---

## Key Decisions

1. **24kHz not 22050Hz** — XTTS v2 outputs at 24000 Hz natively; all extraction normalized to 24kHz to avoid resampling
2. **torchaudio patch is permanent** — `utils/audio_compat.py` must be imported before any torchaudio use in all pipeline scripts
3. **pyannote needs waveform dict** — never pass file paths to pyannote pipeline; always preload with soundfile and pass `{"waveform": ..., "sample_rate": ...}`
4. **Speaker subdirectory structure** — `data/raw/{speaker}/` and `data/audio/{speaker}/` preserved through pipeline for speaker identity
5. **Sequential model loading** — GPU budget: Whisper ~300MB, pyannote ~160MB peak, XTTS ~2.1GB peak; never load two simultaneously

---

## Next Steps

**Immediate — Phase 3.1: Transcription**

1. Write `backend/pipeline/transcribe.py`:

   - Use `faster-whisper large-v3` (fits in ~1.5GB, well within budget)
   - Input: `data/audio/{speaker}/*.wav`
   - Language forced to `ar`
   - Output per file: `data/transcripts/{speaker}/{name}.json` (word-level timestamps) + `.txt`
   - Must `import utils.audio_compat` at top (torchaudio patch)
2. **Phase 3.2 (HUMAN):** Spot-check transcripts for gross errors
3. **Phase 4.1:** `backend/pipeline/diarize.py` — pyannote speaker diarization

   - Pass audio as waveform dict (not file path) — see key decision #3
   - Output: RTTM + simplified JSON
4. **Phase 4.2:** `backend/pipeline/segment.py` — split by speaker, filter 2-30s
5. **Phase 4.3:** Clip into 3-8s training segments

**Notes for transcription script:**

- `faster-whisper` API differs from `openai-whisper`: use `WhisperModel`, returns `(segments, info)` tuple
- Word timestamps: `word_timestamps=True` param
- VRAM: unload model after processing all files (don't keep loaded)

---

## Start Here

```bash
# 1. Verify git state
git log --oneline -5

# 2. Check extraction is complete (should be 23 files, ~2.6GB)
find backend/data/audio -name "*.wav" | wc -l
du -sh backend/data/audio/

# 3. Activate venv and confirm smoke test still passes
backend/.venv/bin/python backend/smoke_test.py
```

---

## Context Notes

- **`dahih` style note:** الدحيح is a fast-paced, energetic Egyptian educational narrator — very different prosody from NatGeo documentary style. Good style contrast for clustering.
- **Memory files** at `memory/` are not git-tracked; contain torchcodec issue notes and project state.
- **PLAN.md phase numbering** is slightly inconsistent with GENERAL_PLAN.md — follow `PLAN.md` (more detailed, active plan).
- The `لهذا السبب قتل الإعلامي السوري محمد السعيد.mp3` file is only 3:56 — borderline for training data; may want to exclude in dataset step.
- Transcription of 15.8h with faster-whisper large-v3 will take ~30-60 min on this GPU.
- **venv activation:** `source backend/.venv/bin/activate` or prefix commands with `backend/.venv/bin/python`
