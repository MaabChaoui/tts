"""
Phase 4.4 — Reorganize clips according to speaker_map.json.

Reads data/speaker_map.json and produces data/clips_final/{narrator_id}/
with sequentially renumbered clips. Source clips are copied (not moved).

Usage:
    python -m pipeline.reorganize
"""

import json
import shutil
import logging
from pathlib import Path

import utils.audio_compat  # noqa: F401

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

BACKEND_DIR = Path(__file__).resolve().parent.parent
CLIPS_DIR = BACKEND_DIR / "data" / "clips"
FINAL_DIR = BACKEND_DIR / "data" / "clips_final"
SPEAKER_MAP = BACKEND_DIR / "data" / "speaker_map.json"


def main():
    mapping = json.loads(SPEAKER_MAP.read_text(encoding="utf-8"))

    for channel, narrators in mapping["channels"].items():
        channel_src = CLIPS_DIR / channel
        for narrator_id, cfg in narrators.items():
            if narrator_id.startswith("_"):
                continue

            sources = cfg["sources"]
            skip_first_n = cfg.get("skip_first_n", 0)
            out_dir = FINAL_DIR / narrator_id
            out_dir.mkdir(parents=True, exist_ok=True)

            # Collect all source clips in sorted order
            all_clips: list[Path] = []
            for spk in sources:
                spk_dir = channel_src / spk
                if not spk_dir.exists():
                    log.warning(f"  Missing: {spk_dir}")
                    continue
                all_clips.extend(sorted(spk_dir.glob("*.wav")))

            # Apply skip_first_n (per-source, so only first source is affected
            # when it's a single-source entry like SPEAKER_06)
            if skip_first_n > 0 and len(sources) == 1:
                skipped = all_clips[:skip_first_n]
                all_clips = all_clips[skip_first_n:]
                log.info(f"  {narrator_id}: skipped first {skip_first_n} clips ({[p.name for p in skipped]})")

            # Copy with sequential renaming
            for idx, src in enumerate(all_clips):
                dst = out_dir / f"clip_{idx:06d}.wav"
                if not dst.exists():
                    shutil.copy2(src, dst)

            log.info(f"  {narrator_id:25s}  {len(all_clips):4d} clips  ← {channel}/{', '.join(sources)}")

    # Summary
    log.info("=" * 60)
    total = 0
    for narrator_dir in sorted(FINAL_DIR.iterdir()):
        if not narrator_dir.is_dir():
            continue
        n = len(list(narrator_dir.glob("*.wav")))
        total += n
        log.info(f"  {narrator_dir.name:25s}  {n:4d} clips")
    log.info(f"  {'TOTAL':25s}  {total:4d} clips")
    log.info(f"Output: {FINAL_DIR}")


if __name__ == "__main__":
    main()
