"""
Phase 6.3 — XTTS v2 GPT fine-tuning.

Fine-tunes only the GPT decoder (encoder + HiFi-GAN frozen) using the
official GPTTrainer from coqui-tts. Tailored for RTX 3050 6GB:
  - batch_size=2, grad_accum=4 → effective batch=8
  - gpt_checkpointing=True (gradient checkpointing inside GPT)
  - fp16 via trainer AMP
  - num_loader_workers=2

Usage:
    python -m training.train [--epochs N] [--batch-size N] [--grad-accum N] [--dry-run]
"""

import argparse
import gc
import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

BACKEND_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BACKEND_DIR / "data"
MODELS_DIR = BACKEND_DIR / "models"
TRAIN_CSV = DATA_DIR / "coqui_train.csv"
EVAL_CSV = DATA_DIR / "coqui_val.csv"

# XTTS v2 model files — downloaded to models/xtts_v2_base/ on first run
XTTS_BASE_DIR = MODELS_DIR / "xtts_v2_base"
XTTS_CHECKPOINT = XTTS_BASE_DIR / "model.pth"
XTTS_CONFIG = XTTS_BASE_DIR / "config.json"
TOKENIZER_FILE = XTTS_BASE_DIR / "vocab.json"
DVAE_CHECKPOINT = XTTS_BASE_DIR / "dvae.pth"
MEL_NORM_FILE = XTTS_BASE_DIR / "mel_stats.pth"

HF_FILES = {
    "model.pth":    "https://huggingface.co/coqui/XTTS-v2/resolve/main/model.pth",
    "config.json":  "https://huggingface.co/coqui/XTTS-v2/resolve/main/config.json",
    "vocab.json":   "https://huggingface.co/coqui/XTTS-v2/resolve/main/vocab.json",
    "dvae.pth":     "https://huggingface.co/coqui/XTTS-v2/resolve/main/dvae.pth",
    "mel_stats.pth":"https://huggingface.co/coqui/XTTS-v2/resolve/main/mel_stats.pth",
}

# Sample sentences for eval during training (one per narrator)
TEST_SENTENCES = [
    {
        "text": "في أعماق المحيطات تكمن أسرار لم يكتشفها الإنسان بعد.",
        "speaker_wav": None,  # filled in at runtime
        "language": "ar",
    },
    {
        "text": "الحرب العالمية الثانية كانت من أشد الصراعات دموية في تاريخ البشرية.",
        "speaker_wav": None,
        "language": "ar",
    },
]


def download_base_model():
    from TTS.utils.manage import ModelManager
    XTTS_BASE_DIR.mkdir(parents=True, exist_ok=True)
    missing = [name for name, url in HF_FILES.items() if not (XTTS_BASE_DIR / name).exists()]
    if missing:
        log.info(f"Downloading {len(missing)} base model file(s): {missing}")
        ModelManager._download_model_files(
            [HF_FILES[name] for name in missing],
            str(XTTS_BASE_DIR),
            progress_bar=True,
        )
    else:
        log.info("Base model files already present.")


def pick_speaker_refs() -> list[dict]:
    """Pick one reference clip per narrator for test_sentences."""
    import random
    random.seed(0)
    refs = []
    clips_final = DATA_DIR / "clips_final"
    for narrator_dir in sorted(clips_final.iterdir()):
        if not narrator_dir.is_dir():
            continue
        wavs = sorted(narrator_dir.glob("*.wav"))
        if wavs:
            refs.append(str(wavs[len(wavs) // 2]))  # pick middle clip
    return refs


def build_config(args):
    from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainerConfig
    from TTS.tts.models.xtts import XttsAudioConfig

    OUT_PATH = str(MODELS_DIR / "finetune" / "run")

    # Pick one speaker ref for test sentences
    speaker_refs = pick_speaker_refs()
    test_sentences = []
    for i, sentence in enumerate(TEST_SENTENCES):
        ref = speaker_refs[i % len(speaker_refs)]
        test_sentences.append({**sentence, "speaker_wav": ref})

    model_args = GPTArgs(
        max_conditioning_length=132300,   # 6 s at 22050 Hz
        min_conditioning_length=66150,    # 3 s at 22050 Hz
        debug_loading_failures=False,
        max_wav_length=int(args.max_audio_len * 22050),
        max_text_length=200,
        mel_norm_file=str(MEL_NORM_FILE),
        dvae_checkpoint=str(DVAE_CHECKPOINT),
        xtts_checkpoint=str(XTTS_CHECKPOINT),
        tokenizer_file=str(TOKENIZER_FILE),
        gpt_num_audio_tokens=1026,
        gpt_start_audio_token=1024,
        gpt_stop_audio_token=1025,
        gpt_use_masking_gt_prompt_approach=True,
        gpt_use_perceiver_resampler=True,
    )

    audio_config = XttsAudioConfig(
        sample_rate=22050,       # GPT conditioning sample rate
        dvae_sample_rate=22050,  # DVAE tokenizer sample rate
        output_sample_rate=24000,
    )

    config = GPTTrainerConfig(
        epochs=args.epochs,
        output_path=OUT_PATH,
        model_args=model_args,
        run_name="XTTS_Arabic_FT",
        project_name="TTS_Arabic",
        dashboard_logger="tensorboard",
        audio=audio_config,
        batch_size=args.batch_size,
        batch_group_size=48,
        eval_batch_size=args.batch_size,
        num_loader_workers=2,
        eval_split_max_size=256,
        print_step=50,
        plot_step=100,
        log_model_step=100,
        save_step=1000,
        save_n_checkpoints=2,
        save_checkpoints=True,
        print_eval=False,
        optimizer="AdamW",
        optimizer_wd_only_on_weights=True,
        optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2},
        lr=5e-6,
        lr_scheduler="MultiStepLR",
        lr_scheduler_params={"milestones": [50000 * 18, 150000 * 18, 300000 * 18], "gamma": 0.5, "last_epoch": -1},
        test_sentences=test_sentences,
    )

    return config, OUT_PATH


def main():
    parser = argparse.ArgumentParser(description="Fine-tune XTTS v2 GPT decoder on Arabic narration data")
    parser.add_argument("--epochs",        type=int,   default=20)
    parser.add_argument("--batch-size",    type=int,   default=2,  dest="batch_size")
    parser.add_argument("--grad-accum",    type=int,   default=4,  dest="grad_accum")
    parser.add_argument("--max-audio-len", type=float, default=8.0, dest="max_audio_len",
                        help="Max audio clip length in seconds (default: 8.0)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Build config and load model only — verify no OOM before full training")
    args = parser.parse_args()

    # Validate input files
    for f in [TRAIN_CSV, EVAL_CSV]:
        if not f.exists():
            raise FileNotFoundError(f"{f} not found — run: python -m training.prepare")

    download_base_model()

    config, out_path = build_config(args)

    from TTS.config.shared_configs import BaseDatasetConfig
    from TTS.tts.datasets import load_tts_samples
    from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTTrainer
    from trainer import Trainer, TrainerArgs

    dataset_config = BaseDatasetConfig(
        formatter="coqui",
        dataset_name="arabic_narration",
        path=str(BACKEND_DIR),       # audio_file paths are relative to here
        meta_file_train=str(TRAIN_CSV),
        meta_file_val=str(EVAL_CSV),
        language="ar",
    )

    log.info("Loading model...")
    model = GPTTrainer.init_from_config(config)

    log.info("Loading dataset samples...")
    train_samples, eval_samples = load_tts_samples(
        [dataset_config],
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )
    log.info(f"  Train: {len(train_samples)}  Eval: {len(eval_samples)}")

    # Count trainable parameters
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Parameters: {total/1e6:.1f}M total, {trainable/1e6:.1f}M trainable ({100*trainable/total:.1f}%)")

    if args.dry_run:
        log.info("Dry run complete — no OOM, ready for full training.")
        log.info("Re-run without --dry-run to start training.")
        return

    trainer = Trainer(
        TrainerArgs(
            restore_path=None,
            skip_train_epoch=False,
            start_with_eval=False,
            grad_accum_steps=args.grad_accum,
        ),
        config,
        output_path=out_path,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )

    log.info(f"Starting training: {args.epochs} epochs, batch={args.batch_size}, grad_accum={args.grad_accum}")
    trainer.fit()

    del model, trainer, train_samples, eval_samples
    gc.collect()

    log.info(f"Training complete. Checkpoints at: {out_path}")


if __name__ == "__main__":
    main()
