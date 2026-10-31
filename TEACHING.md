# TTS Project — Teaching Summary

> What we're building, how the models work, and why each decision was made.

---

## The Goal

Fine-tune a **voice cloning / text-to-speech model** on Arabic documentary narration audio, so we can generate new speech in the style of those narrators — same voice, same cadence, same energy.

Two speaker styles targeted:
- **NatGeo History** — formal, measured, documentary Arabic
- **الدحيح (dahih)** — fast-paced, energetic Egyptian educational style

---

## The Model Stack

### 1. XTTS v2 (Coqui TTS) — *the TTS engine we're fine-tuning*

**What it is:** A multilingual, multi-speaker TTS model based on a transformer architecture.

**How it works:**
- Takes **text** + a **speaker reference audio clip** (~6s) as input
- Encodes the reference clip into a **speaker embedding** — a compact vector (~512 dims) that captures voice identity (pitch distribution, timbre, rhythm)
- A **GPT-2-style autoregressive decoder** generates discrete audio tokens conditioned on the text + speaker embedding
- A **HiFi-GAN vocoder** converts those tokens back to a waveform

**Why fine-tune it?** The base model was trained on many languages including Arabic, but not on *these specific narrators*. Fine-tuning teaches it to replicate their voice and prosodic style.

**Native sample rate:** 24,000 Hz — that's why all our audio is extracted at 24kHz.

---

### 2. faster-whisper large-v3 — *transcription*

**What it is:** OpenAI's Whisper speech recognition model, re-implemented in C++ via CTranslate2 for ~4x faster inference at the same accuracy. `large-v3` is the biggest/most accurate version.

**How it works:**
- Splits audio into 30s windows
- A **CNN encoder** extracts mel-spectrogram features
- A **transformer decoder** autoregressively predicts text tokens
- Trained on 680,000 hours of multilingual audio — Arabic included

**Key settings we use:**
- `language="ar"` — forces Arabic decoding (avoids misdetection)
- `word_timestamps=True` — aligns each word to a time range (needed for segmentation later)
- `vad_filter=True` — Voice Activity Detection pre-filters silence/music before Whisper sees it; reduces hallucinations on non-speech sections and speeds things up

**Performance on our hardware (RTX 3050 6GB):**
- VRAM: ~1.5 GB (fp16)
- Speed: ~0.13x RTF (real-time factor) — 22 min of audio transcribed in ~3 min

---

### 3. pyannote/speaker-diarization-3.1 — *speaker segmentation* (next phase)

**What it is:** A pipeline that answers "who spoke when?"

**How it works:**
- **Segmentation model** (pyannote/segmentation-3.0): a CNN+LSTM that labels each frame as speech/non-speech and detects speaker change points
- **Embedding model**: encodes short audio chunks into speaker-discriminative vectors (like a face embedding but for voices)
- **Clustering**: groups chunks by speaker identity using agglomerative clustering

**Output:** RTTM format — a timestamped list of `(start, end, speaker_id)` turns.

**Why it matters:** Our audio has narrators + sometimes music/background. We need to isolate the narrator's voice segments before building training clips.

---

## Data Pipeline (what we've done and what's coming)

```
Raw MP4/MP3                   ← downloaded from YouTube (you did this)
    ↓  ffmpeg
Mono 24kHz WAV                ← extract.py ✅  (23 files, 15.8h, 2.6 GB)
    ↓  faster-whisper
Transcripts (JSON + TXT)      ← transcribe.py ✅ running now
    ↓  pyannote
Speaker segments (RTTM/JSON)  ← diarize.py  (TODO)
    ↓  pydub
3–8 second clips              ← segment.py  (TODO)
    ↓  parselmouth + sklearn
Prosody clusters              ← cluster.py  (TODO)
    ↓
dataset.csv                   ← dataset.py  (TODO)
    ↓  XTTS fine-tuning
Fine-tuned model              ← train.py    (TODO)
```

---

## Key Technical Decisions

| Decision | Why |
|---|---|
| 24kHz extraction (not 22050Hz) | XTTS v2 outputs at 24kHz natively — no resampling needed |
| faster-whisper over openai-whisper | Same accuracy, ~4x faster, half the VRAM |
| fp16 (half precision) | Halves VRAM usage; GPU arithmetic is still accurate enough for inference |
| VAD filter in Whisper | Documentaries have music/silence — VAD skips those, preventing hallucinated text |
| Sequential model loading | GPU only has 6GB; Whisper + pyannote + XTTS can't coexist — each phase loads then unloads |
| word_timestamps=True | Precise word timings let us align transcripts to audio cuts in segmentation |
| Freeze most of XTTS | Only train speaker embeddings + top layers (~<5% of params) — prevents catastrophic forgetting, keeps training within 6GB |

---

## Concepts Worth Understanding

**Speaker embeddings** — A vector in high-dimensional space where similar voices cluster together. XTTS uses this to condition generation on a target voice. Fine-tuning adds your narrator's voice to this space.

**Transfer learning / partial fine-tuning** — The base XTTS model already "knows" Arabic phonetics and prosody from its training data. We freeze that knowledge and only adapt the parts that encode *who* is speaking. Like teaching someone a new accent without re-teaching them the language.

**Gradient accumulation** — Batch size 1 (one clip at a time) would make training unstable. We accumulate gradients over 8 steps before updating weights — mathematically equivalent to batch size 8, without the VRAM cost.

**Real-Time Factor (RTF)** — Ratio of processing time to audio duration. RTF 0.13x means 1 minute of audio is transcribed in 8 seconds. XTTS inference typically runs at ~0.5–1x RTF on this GPU.

**Prosody** — The musical properties of speech: pitch contour, speaking rate, energy/loudness, pause patterns. Two narrators can say the same sentence with very different prosody. Phase 5 will extract these features and cluster clips by style automatically.

---

## Hardware Constraints Summary

| Model | VRAM | Run alone? |
|---|---|---|
| faster-whisper large-v3 | ~1.5 GB | Yes |
| pyannote diarization | ~1.5 GB | Yes |
| XTTS v2 inference | ~3 GB | Yes |
| XTTS v2 training (frozen, fp16) | ~5–5.5 GB | Yes (tight) |
| Any two simultaneously | >6 GB | **No** |

**Rule:** Each pipeline step loads its model, processes all files, then explicitly `del model` + `torch.cuda.empty_cache()` before the next step starts.
