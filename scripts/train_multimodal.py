from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix, hstack
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
)

DATA_DIR = Path("data/processed/features")
RESULTS_DIR = Path("results")
CHECKPOINT_DIR = Path("checkpoints")

RESULTS_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR.mkdir(exist_ok=True)

# 1. Load aligned metadata
train = pd.read_csv(DATA_DIR / "train_aligned.csv")
dev = pd.read_csv(DATA_DIR / "dev_aligned.csv")

# 2. Load corresponding audio features
X_audio_train = np.load(DATA_DIR / "train_audio.npy")
X_audio_dev = np.load(DATA_DIR / "dev_audio.npy")

assert len(train) == len(X_audio_train)
assert len(dev) == len(X_audio_dev)

# 3. Reuse fitted TF-IDF vectorizer
baseline = joblib.load(CHECKPOINT_DIR / "text_baseline.joblib")
tfidf = baseline.named_steps["tfidf"]

X_text_train = tfidf.transform(train["Utterance"])
X_text_dev = tfidf.transform(dev["Utterance"])

# 4. Standardize audio using training statistics only
scaler = StandardScaler()

X_audio_train = scaler.fit_transform(X_audio_train)
X_audio_dev = scaler.transform(X_audio_dev)

# 5. Concatenate text and audio features
X_train = hstack([
    X_text_train,
    csr_matrix(X_audio_train)
]).tocsr()

X_dev = hstack([
    X_text_dev,
    csr_matrix(X_audio_dev)
]).tocsr()

print("Combined training shape:", X_train.shape)
print("Combined development shape:", X_dev.shape)

# 6. Train multimodal classifier
classifier = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    solver="lbfgs",
)

classifier.fit(X_train, train["Emotion"])

# 7. Evaluate on development data
predictions = classifier.predict(X_dev)

metrics = {
    "accuracy": float(accuracy_score(dev["Emotion"], predictions)),
    "macro_f1": float(
        f1_score(dev["Emotion"], predictions, average="macro")
    ),
    "weighted_f1": float(
        f1_score(dev["Emotion"], predictions, average="weighted")
    ),
}

print("\nMULTIMODAL RESULTS")
print(json.dumps(metrics, indent=2))
print(classification_report(
    dev["Emotion"],
    predictions,
    zero_division=0
))

# 8. Compare against text-only baseline
baseline_predictions = baseline.predict(dev["Utterance"])

print("\nTEXT-ONLY BASELINE")
print("Accuracy:", accuracy_score(
    dev["Emotion"], baseline_predictions
))
print("Macro-F1:", f1_score(
    dev["Emotion"],
    baseline_predictions,
    average="macro"
))

# 9. Save model and preprocessing components
joblib.dump({
    "tfidf": tfidf,
    "audio_scaler": scaler,
    "classifier": classifier,
}, CHECKPOINT_DIR / "multimodal_model.joblib")

with open(RESULTS_DIR / "multimodal_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)