# PLAN.md — Technical Implementation Plan

> Tailored for Claude Code (Opus 4.6) + human collaboration.
> Each task is marked: **[HUMAN]**, **[CLAUDE]**, or **[COLLAB]**.

---

## Project Structure

```
TTS/
├── backend/
│   ├── pipeline/          # data processing scripts
│   │   ├── extract.py     # ffmpeg audio extraction
│   │   ├── transcribe.py  # whisper transcription
│   │   ├── diarize.py     # pyannote speaker segmentation
│   │   ├── segment.py     # split audio by speaker turns
│   │   ├── cluster.py     # prosody-based style clustering
│   │   └── dataset.py     # dataset builder (csv + manifest)
│   ├── training/          # XTTS fine-tuning
│   │   ├── config.py      # training hyperparams + paths
│   │   ├── train.py       # fine-tuning entrypoint
│   │   ├── freeze.py      # layer freezing utilities
│   │   └── eval.py        # generation + listening tests
│   ├── api/               # FastAPI server
│   │   ├── main.py
│   │   ├── routes.py
│   │   └── schemas.py
│   ├── utils/             # shared helpers (audio I/O, logging)
│   ├── data/              # all data artifacts (gitignored)
│   │   ├── raw/           # downloaded videos
│   │   ├── audio/         # extracted wav files
│   │   ├── transcripts/   # whisper output (json + txt)
│   │   ├── segments/      # speaker-split clips
│   │   ├── clips/         # final 3-8s training clips
│   │   └── dataset.csv    # master manifest
│   ├── models/            # saved checkpoints (gitignored)
│   ├── outputs/           # generated audio (gitignored)
│   ├── requirements.txt
│   └── .env               # HF token, paths (gitignored)
├── frontend/              # NextJS app (Phase 2+)
│   └── src/
├── GENERAL_PLAN.md
├── PLAN.md
└── .gitignore
```

---

## Phase 0 — Environment & Setup

### 0.1 Python environment **[CLAUDE]**

- Create `backend/` directory structure
- Create venv: `python -m venv backend/.venv`
- Write `requirements.txt` with pinned versions:

  ```
  torch (CUDA 12.x)
  torchaudio
  TTS              # coqui XTTS v2
  openai-whisper
  pyannote.audio
  faster-whisper   # optional: lower VRAM whisper
  pydub
  librosa
  soundfile
  praat-parselmouth
  scikit-learn
  pandas
  fastapi
  uvicorn
  python-dotenv
  ```
- Validate GPU visibility: `torch.cuda.is_available()`, print VRAM

  ### 0.2 HuggingFace + pyannote access **[HUMAN]**

1. Go to https://huggingface.co/pyannote/speaker-diarization-3.1
2. Accept the user agreement
3. Also accept for https://huggingface.co/pyannote/segmentation-3.0
4. Create a HuggingFace access token (read scope) at https://huggingface.co/settings/tokens
5. Put it in `backend/.env`:
   ```
   HF_TOKEN=hf_...
   ```

### 0.3 Verify tools **[CLAUDE]**

- Write a smoke test script that:
  - Loads Whisper (base model) on a 5s silent wav → confirms it runs
  - Loads pyannote pipeline with HF token → confirms auth works
  - Loads XTTS v2, generates a short Arabic sentence → confirms TTS works
  - Reports VRAM usage at each step
- This tells us exactly what fits in memory simultaneously vs. what must run sequentially

### 0.4 Git init **[CLAUDE]**

- `git init`, write `.gitignore` (data/, models/, outputs/, .venv/, .env, __pycache__)
- Initial commit

---

## Phase 1 — Data Collection

### 1.1 Source selection **[HUMAN]**

- Use `yt-dlp` to download 10-20 NatGeo Abu Dhabi documentary videos
- Prioritize:
  - Long-form (20+ min) documentaries
  - Distinct narrator voices (at least 2-3 recognizable narrators)
  - Minimal background music sections (pure narration preferred)
  - Varied documentary topics (wildlife, history, adventure) for style diversity
- Place files in `backend/data/raw/`
- Create a simple `backend/data/raw/sources.txt` listing video URLs and a short description:
  ```
  https://youtube.com/watch?v=XXX  # WW2 documentary, male narrator
  https://youtube.com/watch?v=YYY  # wildlife, same narrator as XXX
  ```

### 1.2 Download helper (optional) **[CLAUDE]**

- If requested: write a `yt-dlp` wrapper script that reads `sources.txt` and downloads all

---

## Phase 2 — Audio Extraction

### 2.1 Extract & normalize **[CLAUDE]**

Write `backend/pipeline/extract.py`:

- Input: `data/raw/*.mp4`
- ffmpeg: extract audio → mono, 22050 Hz, 16-bit PCM WAV
- Output: `data/audio/{video_name}.wav`
- Log duration per file, total hours collected

**Key decisions:**

- 22050 Hz (XTTS v2 native sample rate, avoids resampling later)
- Mono (narration is mono content; stereo wastes space)

---

## Phase 3 — Transcription

### 3.1 Whisper transcription **[CLAUDE]**

Write `backend/pipeline/transcribe.py`:

- Use `whisper large-v3` (or `faster-whisper large-v3` if VRAM is tight)
- Process each `data/audio/*.wav`
- Output per file:
  - `data/transcripts/{name}.json` — word-level timestamps
  - `data/transcripts/{name}.txt` — plain text
- Language forced to `ar`
- Log WER estimates if available

**VRAM note:** Whisper large-v3 needs ~3-4GB. If running standalone (not alongside XTTS), it fits on 6GB. If `faster-whisper` is used, ~1.5GB.

### 3.2 Transcript review **[HUMAN]**

- Spot-check a few transcripts for gross errors
- No need to fix everything — just flag if Whisper is hallucinating on a specific file
- Mark any files that are mostly music/non-speech for exclusion

---

## Phase 4 — Speaker Diarization & Segmentation

### 4.1 Diarize **[CLAUDE]**

Write `backend/pipeline/diarize.py`:

- Use `pyannote/speaker-diarization-3.1`
- Input: `data/audio/*.wav`
- Output: `data/transcripts/{name}_diarization.rttm` (standard RTTM format)
- Also output a simplified JSON:
  ```json
  [
    {"start": 0.0, "end": 12.5, "speaker": "SPEAKER_00"},
    {"start": 12.5, "end": 30.1, "speaker": "SPEAKER_01"}
  ]
  ```

### 4.2 Segment by speaker **[CLAUDE]**

Write `backend/pipeline/segment.py`:

- Read diarization output + audio
- Cut audio into per-speaker segments
- Filter: discard segments < 2s or > 30s
- Output: `data/segments/{video}_{speaker}_{idx}.wav`

### 4.3 Cut into training clips **[CLAUDE]**

Extend `segment.py` or write `backend/pipeline/clip.py`:

- Take segments and split into **3-8 second clips** (the training max)
- Split on silence/pauses when possible (using `pydub.silence.detect_silence`)
- Discard clips with long silence (>50% silent)
- Output: `data/clips/{video}_{speaker}_{idx}_{clip}.wav`
- Log: total clip count, duration distribution histogram (printed)

### 4.4 Speaker identity mapping **[COLLAB]**

- **[CLAUDE]** generates a report: for each `SPEAKER_XX` across all videos, provide:
  - Number of clips
  - Total duration
  - 3 sample clip paths (for listening)
- **[HUMAN]** listens to samples and assigns human-readable IDs:
  ```
  SPEAKER_00 from video_1 = SPEAKER_02 from video_3 = "narrator_ahmed"
  SPEAKER_01 from video_1 = "narrator_khalid"
  ```
- **[CLAUDE]** writes a `speaker_map.json` and renames/tags clips accordingly

---

## Phase 5 — Style Annotation

### 5.1 Automatic prosody clustering **[CLAUDE]**

Write `backend/pipeline/cluster.py`:

- For each clip, extract features using `parselmouth`:
  - **Pitch**: mean, std, range, contour slope
  - **Energy**: mean RMS, dynamic range
  - **Speaking rate**: syllables/sec (estimated from transcript word count + duration)
  - **Pause ratio**: silence fraction
- Normalize features, run K-Means (k=3 to 6, pick via silhouette score)
- Output:
  - Feature CSV: `data/prosody_features.csv`
  - Cluster assignments with 5 sample clips per cluster
  - Visualization (matplotlib scatter of first 2 PCA components, saved as PNG)

### 5.2 Style labeling **[HUMAN]**

- Listen to sample clips from each cluster
- Assign semantic labels:
  ```
  cluster_0 → "dramatic_intense"
  cluster_1 → "calm_descriptive"
  cluster_2 → "energetic_narration"
  ```
- Write labels in `backend/data/style_labels.json`:
  ```json
  {"0": "dramatic_intense", "1": "calm_descriptive", "2": "energetic_narration"}
  ```

### 5.3 Build master dataset **[CLAUDE]**

Write `backend/pipeline/dataset.py`:

- Combine: clip paths + transcripts + speaker IDs + style labels
- Output `data/dataset.csv`:
  ```csv
  clip_path,transcript,speaker_id,style_tag,duration_s,pitch_mean,energy_mean
  clips/vid1_ahmed_003_01.wav,في أعماق المحيط...,narrator_ahmed,dramatic_intense,5.2,180.3,0.42
  ```
- Validation checks:
  - No missing transcripts
  - No clips outside 3-8s range
  - Speaker/style distribution summary printed
- Train/val split (90/10) stratified by speaker+style

---

## Phase 6 — Model Training (Baseline: Multi-Speaker)

### 6.1 Training config **[CLAUDE]**

Write `backend/training/config.py`:

```python
CONFIG = {
    # Data
    "dataset_csv": "data/dataset.csv",
    "audio_dir": "data/clips/",
    "max_audio_len_s": 8,
    "sample_rate": 22050,

    # Model
    "base_model": "tts_models/multilingual/multi-dataset/xtts_v2",
    "freeze_encoder": True,
    "freeze_decoder": True,       # freeze most decoder layers
    "train_speaker_embeddings": True,
    "train_style_embeddings": False,  # Phase 7 enables this

    # Training
    "batch_size": 1,
    "gradient_accumulation_steps": 8,
    "precision": "fp16",
    "gradient_checkpointing": True,
    "learning_rate": 5e-5,
    "num_epochs": 50,
    "num_workers": 2,
    "save_every_n_epochs": 5,
    "eval_every_n_epochs": 5,

    # Output
    "output_dir": "models/baseline/",
}
```

### 6.2 Layer freezing utility **[CLAUDE]**

Write `backend/training/freeze.py`:

- Load XTTS v2 model
- Freeze all parameters
- Unfreeze only: speaker embedding layer, final projection layers
- Print trainable vs. total parameter count (expect <5% trainable)

### 6.3 Training script **[CLAUDE]**

Write `backend/training/train.py`:

- Load config, dataset, model
- Apply freezing
- Training loop with:
  - fp16 via `torch.cuda.amp`
  - Gradient checkpointing via `torch.utils.checkpoint`
  - Gradient accumulation (manual loop)
  - Logging: loss per step, VRAM usage every N steps
  - Checkpoint saving
- Validation: generate 3 fixed sentences per speaker every eval epoch, save to `outputs/`

### 6.4 Run training **[COLLAB]**

- **[CLAUDE]** prepares everything, does a dry run (1 batch) to verify no OOM
- **[HUMAN]** kicks off full training run, monitors loss
- **[CLAUDE]** writes the eval/listening script

### 6.5 Evaluation **[COLLAB]**

- **[CLAUDE]** writes `backend/training/eval.py`:
  - Load checkpoint, generate audio for N test sentences per speaker
  - Save to `outputs/eval/`
- **[HUMAN]** listens and provides subjective quality feedback:
  - Does it sound like the target narrator?
  - Is pronunciation acceptable?
  - Any artifacts?

---

## Phase 7 — Style Conditioning

### 7.1 Add style embeddings **[CLAUDE]**

- Extend the XTTS model wrapper:
  - Add a `style_embedding` layer (same dim as speaker embeddings)
  - Concatenate or add style embedding to conditioning vector
- Update `config.py`: `train_style_embeddings: True`
- Unfreeze style embedding layer in `freeze.py`

### 7.2 Retrain **[COLLAB]**

- Same flow as Phase 6.4
- **[CLAUDE]** sets up A/B eval: same text, same speaker, different styles
- **[HUMAN]** judges if style differences are audible

---

## Phase 8 — API

### 8.1 FastAPI server **[CLAUDE]**

Write `backend/api/`:

- `POST /generate` — accepts `{text, narrator_id, style_id?}`, returns WAV
- `GET /narrators` — list available narrator IDs
- `GET /styles` — list available style tags
- Model loaded once at startup
- Request validation via Pydantic

### 8.2 Local testing **[COLLAB]**

- **[CLAUDE]** writes curl/httpie test commands
- **[HUMAN]** runs server, tests with real requests

---

## Phase 9 — Frontend

### 9.1 NextJS app **[CLAUDE]**

```
frontend/
├── src/
│   ├── app/
│   │   ├── layout.tsx
│   │   └── page.tsx
│   ├── components/
│   │   ├── NarratorSelect.tsx
│   │   ├── StyleSelect.tsx
│   │   ├── TextInput.tsx
│   │   ├── GenerateButton.tsx
│   │   └── AudioPlayer.tsx
│   └── lib/
│       └── api.ts          # fetch wrapper for backend
├── package.json
├── tailwind.config.ts
└── tsconfig.json
```

- Simple single-page UI: text area, dropdowns, generate button, audio player
- Tailwind for styling
- Calls backend API

---

## Execution Order

| Step | Phase   | Owner  | Depends On | Est. Effort       |
| ---- | ------- | ------ | ---------- | ----------------- |
| 1    | 0.1     | CLAUDE | —         | Setup scripts     |
| 2    | 0.2     | HUMAN  | —         | 10 min            |
| 3    | 0.3     | CLAUDE | 0.1, 0.2   | Smoke tests       |
| 4    | 0.4     | CLAUDE | 0.1        | Git init          |
| 5    | 1.1     | HUMAN  | —         | 1-2 hours         |
| 6    | 2.1     | CLAUDE | 1.1        | Extract script    |
| 7    | 3.1     | CLAUDE | 2.1        | Transcribe script |
| 8    | 3.2     | HUMAN  | 3.1        | 30 min            |
| 9    | 4.1-4.3 | CLAUDE | 3.1        | Pipeline scripts  |
| 10   | 4.4     | COLLAB | 4.3        | Speaker mapping   |
| 11   | 5.1     | CLAUDE | 4.4        | Clustering        |
| 12   | 5.2     | HUMAN  | 5.1        | 30 min            |
| 13   | 5.3     | CLAUDE | 5.2        | Dataset build     |
| 14   | 6.1-6.3 | CLAUDE | 5.3        | Training code     |
| 15   | 6.4     | COLLAB | 6.3        | Training run      |
| 16   | 6.5     | COLLAB | 6.4        | Eval              |
| 17   | 7.1-7.2 | COLLAB | 6.5        | Style training    |
| 18   | 8.1     | CLAUDE | 7.2        | API               |
| 19   | 9.1     | CLAUDE | 8.1        | Frontend          |

---

## GPU Memory Budget (6GB)

| Operation                                  | Est. VRAM | Run Alone? |
| ------------------------------------------ | --------- | ---------- |
| Whisper large-v3                           | ~3.5 GB   | Yes        |
| faster-whisper large-v3                    | ~1.5 GB   | Yes        |
| pyannote diarization                       | ~1.5 GB   | Yes        |
| XTTS v2 inference                          | ~3 GB     | Yes        |
| XTTS v2 training (frozen, fp16, grad ckpt) | ~5-5.5 GB | Yes        |

**Rule: never load two models simultaneously.** Each pipeline step loads its model, processes all files, then unloads before the next step.

---

## Key Concepts Worth Understanding

These are the topics where the learning value is highest — **[HUMAN]** should engage with these rather than just running scripts:

1. **Speaker embeddings** — how XTTS represents different voices as vectors
2. **Transfer learning / partial fine-tuning** — why we freeze most layers and what that means
3. **Gradient accumulation** — simulating larger batches on small GPUs
4. **Mixed precision (fp16)** — what it is, why it halves memory
5. **Speaker diarization** — how pyannote segments multi-speaker audio
6. **Prosody features** — pitch, energy, rate — the building blocks of "style"
7. **Style conditioning** — how adding a learned vector changes generation behavior

I will explain each of these at the relevant phase, not all upfront.
