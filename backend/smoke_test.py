"""
Step 0.3 — Smoke Test Script
Tests Whisper, pyannote, and XTTS v2 sequentially on the 6GB RTX 3050.
Each model is loaded, tested, then fully unloaded before the next.
"""

import gc
import os
import sys
import time
import tempfile
import warnings

import numpy as np
import soundfile as sf
import torch
from dotenv import load_dotenv

# Patch torchaudio to use soundfile (torchcodec has CUDA version mismatch)
sys.path.insert(0, os.path.dirname(__file__))
import utils.audio_compat  # noqa: F401 — patches torchaudio.load/save

# ── Helpers ──────────────────────────────────────────────────────────

ENV_PATH = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(ENV_PATH)

# Auto-accept Coqui TOS for non-interactive runs
os.environ["COQUI_TOS_AGREED"] = "1"


def vram_report(label: str) -> None:
    """Print current VRAM usage."""
    if not torch.cuda.is_available():
        print(f"  [{label}] CUDA not available")
        return
    alloc = torch.cuda.memory_allocated() / 1e6
    reserved = torch.cuda.memory_reserved() / 1e6
    total = torch.cuda.get_device_properties(0).total_memory / 1e6
    print(f"  [{label}] VRAM — allocated: {alloc:.0f} MB | reserved: {reserved:.0f} MB | total: {total:.0f} MB")


def flush_gpu() -> None:
    """Force-free all GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def make_silent_wav(path: str, duration_s: float = 5.0, sr: int = 16000) -> None:
    """Write a silent WAV file for testing."""
    samples = np.zeros(int(sr * duration_s), dtype=np.float32)
    sf.write(path, samples, sr)


def separator(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


# ── Test 1: Whisper ──────────────────────────────────────────────────

def test_whisper() -> bool:
    separator("TEST 1 — Whisper (base model)")
    vram_report("before load")

    try:
        import whisper

        t0 = time.time()
        model = whisper.load_model("base", device="cuda")
        print(f"  Loaded whisper-base in {time.time() - t0:.1f}s")
        vram_report("after load")

        # Transcribe a 5s silent WAV
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_wav = f.name
        make_silent_wav(tmp_wav)

        result = model.transcribe(tmp_wav, language="ar", fp16=True)
        print(f"  Transcription result: '{result['text'].strip()}'")
        print(f"  (empty or near-empty is expected for silence)")
        vram_report("after transcribe")

        os.unlink(tmp_wav)
        del model
        flush_gpu()
        vram_report("after unload")
        print("\n  >>> Whisper: PASSED")
        return True

    except Exception as e:
        print(f"\n  >>> Whisper: FAILED — {e}")
        flush_gpu()
        return False


# ── Test 2: pyannote ─────────────────────────────────────────────────

def test_pyannote() -> bool:
    separator("TEST 2 — pyannote (speaker-diarization-3.1)")
    vram_report("before load")

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("  >>> pyannote: FAILED — HF_TOKEN not set in .env")
        return False

    try:
        from pyannote.audio import Pipeline

        t0 = time.time()
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=hf_token,
        )
        pipeline.to(torch.device("cuda"))
        print(f"  Loaded pyannote pipeline in {time.time() - t0:.1f}s")
        vram_report("after load")

        # Diarize a 5s silent WAV
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_wav = f.name
        make_silent_wav(tmp_wav, sr=16000)

        diarization = pipeline(tmp_wav)
        n_speakers = len(set(label for _, _, label in diarization.itertracks(yield_label=True)))
        print(f"  Speakers detected in silence: {n_speakers} (0 expected)")
        vram_report("after diarize")

        os.unlink(tmp_wav)
        del pipeline, diarization
        flush_gpu()
        vram_report("after unload")
        print("\n  >>> pyannote: PASSED")
        return True

    except Exception as e:
        print(f"\n  >>> pyannote: FAILED — {e}")
        flush_gpu()
        return False


# ── Test 3: XTTS v2 ─────────────────────────────────────────────────

def test_xtts() -> bool:
    separator("TEST 3 — XTTS v2 (Arabic TTS)")
    vram_report("before load")

    try:
        from TTS.api import TTS

        t0 = time.time()
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)
        print(f"  Loaded XTTS v2 in {time.time() - t0:.1f}s")
        vram_report("after load")

        # We need a short reference audio clip for voice cloning.
        # Generate a 3s silent clip as a placeholder reference.
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            ref_wav = f.name
        # XTTS needs a non-silent reference to extract speaker embedding.
        # Use a tiny sine wave so it has something to work with.
        sr = 22050
        t = np.linspace(0, 3.0, int(sr * 3.0), dtype=np.float32)
        tone = 0.3 * np.sin(2 * np.pi * 220 * t)  # 220 Hz tone
        sf.write(ref_wav, tone, sr)

        out_path = os.path.join(tempfile.gettempdir(), "smoke_test_xtts.wav")
        t0 = time.time()
        tts.tts_to_file(
            text="هذا اختبار لنظام تحويل النص إلى كلام",
            file_path=out_path,
            speaker_wav=ref_wav,
            language="ar",
        )
        dur = time.time() - t0
        print(f"  Generated Arabic speech in {dur:.1f}s")
        vram_report("after generate")

        # Verify output exists and has content
        if os.path.exists(out_path):
            info = sf.info(out_path)
            print(f"  Output: {info.duration:.2f}s, {info.samplerate} Hz, {info.channels} ch")
            os.unlink(out_path)
        else:
            print("  WARNING: output file not created")

        os.unlink(ref_wav)
        del tts
        flush_gpu()
        vram_report("after unload")
        print("\n  >>> XTTS v2: PASSED")
        return True

    except Exception as e:
        print(f"\n  >>> XTTS v2: FAILED — {e}")
        import traceback
        traceback.print_exc()
        flush_gpu()
        return False


# ── Main ─────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  TTS Project — Smoke Test (Step 0.3)")
    print(f"  Python {sys.version.split()[0]} | PyTorch {torch.__version__}")
    print(f"  CUDA: {torch.cuda.is_available()} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("\nFATAL: CUDA not available. Aborting.")
        sys.exit(1)

    # Suppress noisy warnings during smoke test
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    results = {}
    results["whisper"] = test_whisper()
    results["pyannote"] = test_pyannote()
    results["xtts"] = test_xtts()

    # ── Summary ──
    separator("SUMMARY")
    all_pass = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  {name:12s} : {status}")

    print()
    if all_pass:
        print("  All smoke tests passed! Ready for Phase 1.")
    else:
        print("  Some tests failed. Check output above for details.")
    print("=" * 60)
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
