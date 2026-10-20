from pathlib import Path
import json

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.pipeline import Pipeline
import joblib

train = pd.read_csv("data/processed/train_manifest.csv")
dev = pd.read_csv("data/processed/dev_manifest.csv")

model = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
    )),
])

model.fit(train["Utterance"], train["Emotion"])

predictions = model.predict(dev["Utterance"])

metrics = {
    "accuracy": float(accuracy_score(dev["Emotion"], predictions)),
    "macro_f1": float(f1_score(
        dev["Emotion"], predictions, average="macro"
    )),
    "weighted_f1": float(f1_score(
        dev["Emotion"], predictions, average="weighted"
    )),
}

print(json.dumps(metrics, indent=2))
print(classification_report(dev["Emotion"], predictions, zero_division=0))

Path("checkpoints").mkdir(exist_ok=True)
Path("results").mkdir(exist_ok=True)

joblib.dump(model, "checkpoints/text_baseline.joblib")

with open("results/text_baseline_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)