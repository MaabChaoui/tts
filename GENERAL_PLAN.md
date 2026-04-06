# GENERAL_PLAN.md

## 🎯 Project Overview

This project aims to build a **multi-style Arabic documentary narration engine** capable of transforming arbitrary text into speech resembling iconic narration styles (e.g., WW2 narrator, wildlife narrator, Wicked Tuna).

This is a **hobby / learning project** with two goals:

* Learn advanced AI (TTS, style modeling, data pipelines)
* Build a fun system for generating high-quality meme/documentary-style audio

The system will be built incrementally in  **phases** , starting simple and adding complexity.

Local machine specs: 

* rtx 3050 with 6gb of VRAM
* 16gb RAM
* Intel i5-13th gen
* kernel: Linux 6.19.10-arch1-1

---

# 🧱 Core System Definition

Target function:

```
(text, narrator_id, style_id, intensity) → audio
```

MVP starts simpler:

```
(text, narrator_id) → audio
```

Then evolves to:

```
(text, narrator_id, style_id) → audio
```

---

# ⚙️ Tech Stack (Initial)

### Backend

* FastAPI
* Python (main_env)

### Models

* XTTS v2 (primary TTS base)
* Whisper (transcription)

### Tools

* ffmpeg (audio extraction)
* pyannote (speaker diarization)
* Praat / Parselmouth (prosody features)

### Frontend (later)

* Minimal UI (Next.js or simple HTML)

---

# 📦 DATA PIPELINE (CRITICAL)

## Phase 0 — Data Collection

### Goal

Collect **5–10 hours minimum** of clean narration audio.

### Sources

* NatGeo Abu Dhabi YouTube videos
* Prefer:
  * Single narrator segments
  * Clean background (minimal music)

### Output

```
data/raw/
    video_1.mp4
    video_2.mp4
```

---

## Phase 1 — Audio Extraction

### Steps

* Extract audio using ffmpeg
* Convert to:
  * mono
  * 16kHz or 22kHz WAV

### Output

```
data/audio/
    file_1.wav
    file_2.wav
```

---

## Phase 2 — Transcription

### Steps

* Use Whisper (large-v3)
* Generate Arabic transcripts

### Output

```
data/transcripts/
    file_1.txt
    file_2.txt
```

⚠️ Notes:

* Arabic punctuation will be weak
* Accept imperfect transcripts for MVP

---

## Phase 3 — Speaker Segmentation

### Approach

Use:

* pyannote diarization

### Goal

Split audio into **single-speaker segments**

### Output

```
data/segments/
    segment_001.wav
    segment_002.wav
```

---

## Phase 4 — Alignment (Optional for MVP)

Skip for MVP if needed.

Later:

* forced alignment (text ↔ audio)

---

## Phase 5 — Style Annotation (OPTION 3: Hybrid)

### Step 1 — Automatic Clustering

Cluster segments using:

* pitch variance
* speaking rate
* energy

### Step 2 — Manual Labeling

Assign labels to clusters:

Example:

```
dramatic_historical
calm_wildlife
energetic_adventure
```

### Output dataset format:

```
dataset.csv
```

```csv
audio_path,text,speaker_id,style_tag
segment_001.wav,...,ww2_narrator,dramatic_historical
segment_045.wav,...,tuna_narrator,energetic_adventure
```

---

# 🧠 MODEL DEVELOPMENT

---

## Phase 1 — Baseline (Multi-Speaker Only)

### Goal

Train:

```
(text, speaker_id) → audio
```

### Steps

* Fine-tune XTTS v2
* Use speaker_id only (ignore style)

### Expected Result

* Different narrator voices
* Limited style control

---

## Phase 2 — Add Style Conditioning (OPTION B)

### Goal

Train:

```
(text, speaker_id, style_id) → audio
```

### Method

* Introduce **style embeddings (tokens)**
* Each style_tag → embedding vector

### Result

* Same narrator, multiple styles
* Different narrators, shared styles

---

## Phase 3 — Style Control Improvements

Add:

* style intensity parameter
* better balancing between styles

Optional:

* contrastive loss (to separate styles)

---

# 🧪 TRAINING STRATEGY

## General Rules

* Start small (subset of data)
* Train short runs first
* Validate frequently

---

## Data Balancing

⚠️ Important:

* Ensure no narrator dominates
* Ensure style distribution is balanced

---

## Evaluation

### Metrics (informal)

* Does it sound like the narrator?
* Does style match expectation?
* Does it generalize to new text?

---

# 🌐 API DESIGN (MVP)

## Phase 1

```
POST /generate
{
  "text": "...",
  "narrator": "ww2_narrator"
}
```

---

## Phase 2

```
POST /generate
{
  "text": "...",
  "narrator": "ww2_narrator",
  "style": "dramatic_historical"
}
```

---

# 🖥️ UI (Optional Early Version)

Simple interface:

* Text input
* Narrator dropdown
* Style dropdown (later)
* Generate button
* Audio player

---

# 🚀 PHASE ROADMAP

---

## ✅ Phase 0 — Setup

* Environment (main_env)
* Install dependencies
* Test Whisper + XTTS locally

---

## ✅ Phase 1 — Data Pipeline MVP

* Extract audio
* Transcribe
* Segment speakers
* Build initial dataset

---

## ✅ Phase 2 — Baseline Model

* Fine-tune XTTS (multi-speaker)
* Build simple API
* Generate first outputs

---

## ✅ Phase 3 — Style System

* Cluster styles
* Label dataset
* Add style embeddings
* Retrain model

---

## 🔜 Phase 4 — Improvements

* Better transcription (punctuation)
* Prosody modeling
* Style intensity control

---

## 🔜 Phase 5 — Advanced (Optional)

* Style interpolation
* Retrieval-based prosody
* LLM-based text rewriting

---

# ⚠️ RISKS & PITFALLS

### 1. Noisy Data

* Background music
* Multiple speakers

→ mitigate via segmentation

---

### 2. Style Confusion

* Model mixes styles

→ mitigate via:

* clean labels
* balanced dataset

---

### 3. Overfitting

* Sounds good on training text only

→ test on unseen text

---

### 4. Arabic Challenges

* Weak punctuation
* phoneme inconsistencies

→ improve later

---

# 🧠 DEVELOPMENT PHILOSOPHY

* Start simple → iterate
* Don’t over-engineer early
* Validate at every phase
* Keep everything reproducible
* Log everything (configs, outputs)

---

# 🎯 FINAL NOTE

This project is:

* A **learning platform**
* A **creative tool**
* A **foundation for more advanced AI systems**

Focus on:

* getting something working early
* then improving quality step by step

---

# ✅ NEXT STEP

Start with:

→ Phase 0 + Phase 1
→ Build first dataset
→ Generate first audio ASAP

Then iterate.
