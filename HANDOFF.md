# HANDOFF — TTS Project (Phase 0.1 Complete)

## Project Overview
- **Objective**: Fine-tune XTTS v2 for high-quality Arabic narration using NatGeo Abu Dhabi documentary styles.
- **Architecture**: Modular backend (Pipeline → Training → API) with a Phase 2 Next.js frontend.
- **Tech Stack**: Python 3.11 (via `uv`), PyTorch 2.11+cu128, `coqui-tts` (Idiap fork), `pyannote.audio` 4.0.4, `openai-whisper`.
- **Reference**: `PLAN.md` (Implementation steps) and `GENERAL_PLAN.md` (High-level goals).

## Files Modified This Session
- `backend/requirements.txt`: Curated dependency list with specific pins for `transformers` and `pyannote`.
- `backend/.env`: Environment template with `HF_TOKEN` placeholder.
- `backend/validate_gpu.py`: GPU/Torch/VRAM validation script (RTX 3050 6GB verified).
- `backend/pipeline/`, `training/`, `api/`, `utils/`: Created directory skeletons with `__init__.py`. 
- `backend/data/`, `models/`, `outputs/`: Data artifact directory structure (gitignored).

## Current Status
- **Phase 0.1 (Setup) is 100% DONE.**
- Environment is fully provisioned in `backend/.venv` using `uv`.
- **RESOLVED**: Major `ImportError` on `transformers` utilities by pinning to **`4.53.0`**.
- **RESOLVED**: `AttributeError` on `torchaudio` by upgrading to **`pyannote.audio==4.0.4`**.
- **RESOLVED**: Missing codec error by adding **`torchcodec==0.11.0`**.

## Key Decisions
- **Python 3.11.14**: Chosen as the "Goldilocks" version for best compatibility with legacy ML frameworks and modern PyTorch.
- **`uv` over `venv`**: Used for faster provisioning and better handling of multiple Python side-by-side.
- **Community Fork**: Explicitly used `coqui-tts` (Idiap fork) as the original Coqui AI repo is unmaintained.

## Next Steps
- **0.2 [HUMAN]**: Obtain HuggingFace token and accept `pyannote` user agreements.
- **0.3 [CLAUDE]**: Write the full smoke test script (Phase 0.3) to verify inference (Whisper/XTTS/Pyannote) and VRAM fit.
- **1.1 [HUMAN]**: Populate `backend/data/raw/` with documentary samples via `yt-dlp`.

## Start Here
1. `cat PLAN.md` (Review execution status table).
2. `backend/.venv/bin/python backend/validate_gpu.py` (Verify GPU/Torch state).
3. `backend/.venv/bin/python -c "import TTS; import pyannote.audio; print('Stack OK')"` (Final import check).

## Context Notes
- **VRAM limit**: The GPU is an RTX 3050 (6GB). We MUST run pipeline steps sequentially to avoid OOM.
- **Semantic Note**: "coqui-tts" on PyPI is the correct package; the original "TTS" is deprecated.
- **Pruning**: `backend/validate_gpu.py` can be deleted once 0.3 smoke tests are done.
