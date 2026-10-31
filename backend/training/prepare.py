"""
Phase 6 prep — Convert dataset.csv to coqui pipe-delimited format.

The coqui formatter expects:
  audio_file|text|speaker_name|emotion_name

Where audio_file is relative to the root_path (BACKEND_DIR).
style_tag → emotion_name column.

Outputs:
  data/coqui_train.csv
  data/coqui_val.csv

Usage:
    python -m training.prepare
"""

import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

BACKEND_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BACKEND_DIR / "data"


def convert(src_csv: Path, dst_csv: Path):
    import pandas as pd

    df = pd.read_csv(src_csv)

    # Drop rows with empty transcripts
    before = len(df)
    df = df[df["transcript"].notna() & (df["transcript"].str.strip() != "")]
    after = len(df)
    if before != after:
        log.warning(f"  Dropped {before - after} rows with empty transcripts")

    # Build coqui format
    coqui = []
    for _, row in df.iterrows():
        coqui.append({
            "audio_file": row["clip_path"],        # relative to BACKEND_DIR
            "text": row["transcript"].strip(),
            "speaker_name": row["narrator_id"],
            "emotion_name": row["style_tag"],
        })

    import csv
    with open(dst_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["audio_file", "text", "speaker_name", "emotion_name"], delimiter="|")
        writer.writeheader()
        writer.writerows(coqui)

    log.info(f"  {src_csv.name} → {dst_csv.name}  ({len(coqui)} rows)")


def main():
    convert(DATA_DIR / "dataset_train.csv", DATA_DIR / "coqui_train.csv")
    convert(DATA_DIR / "dataset_val.csv",   DATA_DIR / "coqui_val.csv")
    log.info("Done. Ready for training.")


if __name__ == "__main__":
    main()
