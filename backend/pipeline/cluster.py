"""
Phase 5.1 — Prosody feature extraction + K-Means style clustering.

For each clip in data/clips_final/, extracts:
  - Pitch: mean, std, range, slope
  - Energy: mean, std (via parselmouth intensity)
  - Pause ratio: fraction of low-energy frames
  - ZCR: zero-crossing rate (speaking rate proxy)

Then runs K-Means (k=3..6), picks best k via silhouette score, and outputs:
  - data/prosody_features.csv  — full feature table + cluster assignment
  - data/cluster_samples.txt   — 5 sample clips per cluster (for human review)
  - data/pca_clusters.png      — 2D PCA scatter plot

Usage:
    python -m pipeline.cluster
"""

import logging
import warnings
from pathlib import Path

import utils.audio_compat  # noqa: F401

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

BACKEND_DIR = Path(__file__).resolve().parent.parent
CLIPS_FINAL_DIR = BACKEND_DIR / "data" / "clips_final"
DATA_DIR = BACKEND_DIR / "data"


def extract_features(wav_path: Path) -> dict | None:
    import parselmouth
    import numpy as np
    import librosa

    try:
        sound = parselmouth.Sound(str(wav_path))
        duration = sound.duration

        # --- Pitch ---
        pitch = sound.to_pitch()
        pitch_vals = pitch.selected_array["frequency"]
        voiced = pitch_vals[pitch_vals > 0]
        if len(voiced) >= 5:
            pitch_mean = float(np.mean(voiced))
            pitch_std = float(np.std(voiced))
            pitch_range = float(np.max(voiced) - np.min(voiced))
            t = np.linspace(0, 1, len(voiced))
            pitch_slope = float(np.polyfit(t, voiced, 1)[0])
            voiced_ratio = float(len(voiced) / len(pitch_vals)) if len(pitch_vals) > 0 else 0.0
        else:
            pitch_mean = pitch_std = pitch_range = pitch_slope = 0.0
            voiced_ratio = 0.0

        # --- Intensity / Energy ---
        intensity = sound.to_intensity()
        int_vals = intensity.values[0]
        energy_mean = float(np.mean(int_vals))
        energy_std = float(np.std(int_vals))

        # Pause ratio: frames more than 1 std below mean
        pause_ratio = float(np.mean(int_vals < (energy_mean - energy_std)))

        # --- ZCR (speaking rate proxy) ---
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y, sr = librosa.load(str(wav_path), sr=None, mono=True)
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))

        return {
            "clip_path": str(wav_path.relative_to(BACKEND_DIR)),
            "narrator_id": wav_path.parent.name,
            "duration_s": round(duration, 3),
            "pitch_mean": round(pitch_mean, 2),
            "pitch_std": round(pitch_std, 2),
            "pitch_range": round(pitch_range, 2),
            "pitch_slope": round(pitch_slope, 2),
            "voiced_ratio": round(voiced_ratio, 4),
            "energy_mean": round(energy_mean, 2),
            "energy_std": round(energy_std, 2),
            "pause_ratio": round(pause_ratio, 4),
            "zcr": round(zcr, 6),
        }
    except Exception as e:
        log.warning(f"  Feature extraction failed for {wav_path.name}: {e}")
        return None


def run_clustering(df, feature_cols: list[str]):
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    X = df[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA for visualization
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X_scaled)
    log.info(f"PCA explained variance: {pca.explained_variance_ratio_.round(3)}")

    # Find best k
    best_k, best_score, best_labels = 3, -1, None
    for k in range(3, 7):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        log.info(f"  k={k}  silhouette={score:.4f}")
        if score > best_score:
            best_k, best_score, best_labels = k, score, labels

    log.info(f"Best k={best_k} (silhouette={best_score:.4f})")
    return best_labels, X_2d, best_k


def main():
    import pandas as pd
    import numpy as np
    import random

    random.seed(42)

    # --- Feature extraction ---
    wav_files = sorted(CLIPS_FINAL_DIR.rglob("*.wav"))
    log.info(f"Extracting features from {len(wav_files)} clips...")

    rows = []
    for i, wav in enumerate(wav_files):
        if i % 200 == 0:
            log.info(f"  {i}/{len(wav_files)}")
        feat = extract_features(wav)
        if feat:
            rows.append(feat)

    import pandas as pd
    df = pd.DataFrame(rows)
    log.info(f"Extracted features for {len(df)} clips ({len(wav_files)-len(df)} failed)")

    # --- Clustering ---
    feature_cols = ["pitch_mean", "pitch_std", "pitch_range", "pitch_slope",
                    "voiced_ratio", "energy_mean", "energy_std", "pause_ratio", "zcr"]

    log.info("Running K-Means clustering (k=3..6)...")
    labels, X_2d, best_k = run_clustering(df, feature_cols)

    df["cluster"] = labels
    df["pca_x"] = X_2d[:, 0]
    df["pca_y"] = X_2d[:, 1]

    # --- Save CSV ---
    csv_path = DATA_DIR / "prosody_features.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")
    log.info(f"Saved: {csv_path}")

    # --- Cluster samples report ---
    import numpy as np
    samples_path = DATA_DIR / "cluster_samples.txt"
    lines = [f"Best k={best_k}\n"]
    for c in range(best_k):
        cluster_df = df[df["cluster"] == c]
        narrator_dist = cluster_df["narrator_id"].value_counts().to_dict()
        lines.append(f"\n--- Cluster {c} ({len(cluster_df)} clips) ---")
        lines.append(f"Narrator distribution: {narrator_dist}")
        lines.append(f"pitch_mean={cluster_df['pitch_mean'].mean():.1f}  "
                     f"energy_mean={cluster_df['energy_mean'].mean():.1f}  "
                     f"pause_ratio={cluster_df['pause_ratio'].mean():.3f}  "
                     f"zcr={cluster_df['zcr'].mean():.5f}")
        samples = cluster_df["clip_path"].sample(min(5, len(cluster_df)), random_state=42).tolist()
        lines.append("Samples:")
        for s in samples:
            lines.append(f"  {s}")

    samples_path.write_text("\n".join(lines), encoding="utf-8")
    log.info(f"Saved: {samples_path}")

    # --- PCA plot ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Left: colored by cluster
        scatter = axes[0].scatter(X_2d[:, 0], X_2d[:, 1], c=df["cluster"],
                                   cmap="tab10", alpha=0.5, s=8)
        axes[0].set_title(f"PCA — Style Clusters (k={best_k})")
        axes[0].set_xlabel("PC1"); axes[0].set_ylabel("PC2")
        plt.colorbar(scatter, ax=axes[0], label="cluster")

        # Right: colored by narrator
        narrator_ids = df["narrator_id"].unique()
        cmap2 = plt.cm.get_cmap("Set1", len(narrator_ids))
        for i, nid in enumerate(sorted(narrator_ids)):
            mask = df["narrator_id"] == nid
            axes[1].scatter(X_2d[mask, 0], X_2d[mask, 1],
                            color=cmap2(i), alpha=0.4, s=8, label=nid)
        axes[1].set_title("PCA — Narrator Identity")
        axes[1].set_xlabel("PC1"); axes[1].set_ylabel("PC2")
        axes[1].legend(markerscale=3, fontsize=8)

        plt.tight_layout()
        plot_path = DATA_DIR / "pca_clusters.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        log.info(f"Saved: {plot_path}")
    except Exception as e:
        log.warning(f"Plot failed: {e}")

    # --- Final summary ---
    log.info("=" * 60)
    for c in range(best_k):
        n = (df["cluster"] == c).sum()
        log.info(f"  Cluster {c}: {n} clips")
    log.info(f"Total: {len(df)} clips in {best_k} style clusters")


if __name__ == "__main__":
    main()
