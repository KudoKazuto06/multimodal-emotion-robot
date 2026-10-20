
"""Benchmark local multimodal emotion inference latency."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.inference import EmotionPredictor


# Configuration
MANIFEST_PATH = Path("data/processed/dev_manifest.csv")
RESULTS_DIR = Path("results")
OUTPUT_PATH = RESULTS_DIR / "latency_benchmark.json"

N_SAMPLES = 100
N_WARMUP = 5


def main():
    # Create results directory
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load development data
    df = pd.read_csv(MANIFEST_PATH)

    if df.empty:
        raise ValueError("Development dataset is empty")

    n_samples = min(N_SAMPLES, len(df))

    print(f"Loaded {len(df)} development examples")
    print(f"Benchmarking {n_samples} samples")

    # Load model once, outside timing
    predictor = EmotionPredictor()

    # Warm-up runs
    print("\nWarming up model...")

    warmup = df.iloc[0]

    for _ in range(N_WARMUP):
        predictor.predict(
            text=str(warmup["Utterance"]),
            audio_path=warmup["audio_path"]
        )

    # Benchmark inference
    print("\nRunning benchmark...")

    latencies = []

    for _, row in tqdm(
        df.head(n_samples).iterrows(),
        total=n_samples,
        desc="Benchmarking"
    ):
        result = predictor.predict(
            text=str(row["Utterance"]),
            audio_path=row["audio_path"]
        )

        latencies.append(result["latency_ms"])

    # Calculate latency statistics
    latencies = np.asarray(latencies, dtype=np.float64)

    metrics = {
        "samples": int(len(latencies)),
        "warmup_runs": N_WARMUP,
        "mean_ms": round(float(np.mean(latencies)), 2),
        "median_ms": round(float(np.median(latencies)), 2),
        "p95_ms": round(float(np.percentile(latencies, 95)), 2),
        "min_ms": round(float(np.min(latencies)), 2),
        "max_ms": round(float(np.max(latencies)), 2),
        "target_latency_ms": 100,
        "within_target_percent": round(
            float(np.mean(latencies <= 100) * 100), 2
        ),
        "measurement_scope": (
            "Local inference including audio file loading, "
            "MFCC feature extraction, TF-IDF transformation, "
            "feature scaling, emotion classification, "
            "and rule-based response generation."
        ),
        "excluded_from_timing": [
            "Model loading",
            "Microphone capture",
            "User speech duration",
            "UI rendering"
        ]
    }

    # Display results
    print("\n=== INFERENCE BENCHMARK RESULTS ===")
    print(json.dumps(metrics, indent=2))

    # Save results
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nResults saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
