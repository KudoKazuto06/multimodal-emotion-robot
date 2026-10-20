
"""Profile model size and local inference resource usage."""

import json
import platform
import subprocess
from pathlib import Path

import joblib
import psutil
import pandas as pd

from src.inference import EmotionPredictor


MODEL_PATH = Path("checkpoints/multimodal_model.joblib")
MANIFEST_PATH = Path("data/processed/dev_manifest.csv")
OUTPUT_PATH = Path("results/resource_profile.json")


def get_gpu_info():
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "Unavailable"


def main():
    bundle = joblib.load(MODEL_PATH)

    tfidf = bundle["tfidf"]
    scaler = bundle["audio_scaler"]
    classifier = bundle["classifier"]

    # Learned classifier coefficients and intercepts
    classifier_parameters = (
        classifier.coef_.size + classifier.intercept_.size
    )

    # Stored fitted preprocessing values
    tfidf_idf_values = tfidf.idf_.size

    scaler_values = (
        scaler.mean_.size
        + scaler.var_.size
        + scaler.scale_.size
    )

    # Total stored learned/fitted numerical values
    total_fitted_values = (
        classifier_parameters
        + tfidf_idf_values
        + scaler_values
    )

    # Checkpoint size
    checkpoint_mb = MODEL_PATH.stat().st_size / (1024 ** 2)

    # Load a real development utterance
    df = pd.read_csv(MANIFEST_PATH)
    sample = df.iloc[0]

    process = psutil.Process()

    # Measure process memory before loading predictor
    ram_before_mb = process.memory_info().rss / (1024 ** 2)

    predictor = EmotionPredictor()

    ram_loaded_mb = process.memory_info().rss / (1024 ** 2)

    # Warm-up
    predictor.predict(
        str(sample["Utterance"]),
        sample["audio_path"]
    )

    ram_after_mb = process.memory_info().rss / (1024 ** 2)

    profile = {
        "hardware": {
            "cpu": platform.processor(),
            "cpu_logical_cores": psutil.cpu_count(logical=True),
            "total_system_ram_gb": round(
                psutil.virtual_memory().total / (1024 ** 3), 2
            ),
            "gpu": get_gpu_info(),
        },
        "model": {
            "architecture": "TF-IDF + MFCC + Logistic Regression",
            "text_features": len(tfidf.vocabulary_),
            "audio_features": int(scaler.n_features_in_),
            "classifier_coefficients_and_intercepts":
                int(classifier_parameters),
            "idf_values": int(tfidf_idf_values),
            "scaler_stored_values": int(scaler_values),
            "total_fitted_numeric_values":
                int(total_fitted_values),
            "checkpoint_mb": round(checkpoint_mb, 3),
        },
        "memory": {
            "process_ram_before_model_load_mb":
                round(ram_before_mb, 2),
            "process_ram_after_model_load_mb":
                round(ram_loaded_mb, 2),
            "process_ram_after_warmup_mb":
                round(ram_after_mb, 2),
        },
        "runtime": {
            "python": platform.python_version(),
            "inference_device": "CPU",
            "requires_cuda": False,
        },
    }

    OUTPUT_PATH.parent.mkdir(exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)

    print(json.dumps(profile, indent=2))
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
