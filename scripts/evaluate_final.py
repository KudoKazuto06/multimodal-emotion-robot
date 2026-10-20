
"""Final held-out evaluation of text-only and multimodal models."""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix, hstack
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay,
)

DATA_DIR = Path("data/processed/features")
CHECKPOINT_DIR = Path("checkpoints")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# Load test data
test = pd.read_csv(DATA_DIR / "test_aligned.csv")
audio = np.load(DATA_DIR / "test_audio.npy")

assert len(test) == len(audio), "Audio and text rows do not match"

# Load trained models
text_model = joblib.load(
    CHECKPOINT_DIR / "text_baseline.joblib"
)
bundle = joblib.load(
    CHECKPOINT_DIR / "multimodal_model.joblib"
)

tfidf = bundle["tfidf"]
scaler = bundle["audio_scaler"]
classifier = bundle["classifier"]

y_true = test["Emotion"]

# Text-only predictions
text_predictions = text_model.predict(test["Utterance"])

# Multimodal predictions
text_features = tfidf.transform(test["Utterance"])
audio_features = scaler.transform(audio)

combined = hstack([
    text_features,
    csr_matrix(audio_features)
]).tocsr()

multimodal_predictions = classifier.predict(combined)


def evaluate(name, predictions):
    metrics = {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "macro_f1": float(f1_score(
            y_true, predictions, average="macro"
        )),
        "weighted_f1": float(f1_score(
            y_true, predictions, average="weighted"
        )),
    }

    print(f"\n=== {name} ===")
    print(json.dumps(metrics, indent=2))
    print(classification_report(
        y_true, predictions, zero_division=0
    ))

    return metrics


results = {
    "test_samples": len(test),
    "text_only": evaluate(
        "TEXT ONLY", text_predictions
    ),
    "multimodal": evaluate(
        "TEXT + AUDIO", multimodal_predictions
    ),
}

results["difference"] = {
    key: results["multimodal"][key] - results["text_only"][key]
    for key in ["accuracy", "macro_f1", "weighted_f1"]
}

# Save results
with open(
    RESULTS_DIR / "final_test_metrics.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(results, f, indent=2)

# Generate confusion matrix
fig, ax = plt.subplots(figsize=(9, 7))

ConfusionMatrixDisplay.from_predictions(
    y_true,
    multimodal_predictions,
    labels=classifier.classes_,
    xticks_rotation=45,
    cmap="Blues",
    ax=ax,
    colorbar=False,
    values_format="d",
)

ax.set_title("Multimodal Emotion Classification — MELD Test Set")
fig.tight_layout()

fig.savefig(
    RESULTS_DIR / "confusion_matrix.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close(fig)

print("\nFinal results saved to results/")
