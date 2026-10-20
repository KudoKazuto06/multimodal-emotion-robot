"""Local Text + Audio emotion inference for the MELD prototype."""

from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import librosa
import numpy as np
from scipy.sparse import csr_matrix, hstack

from src.response import generate_response


SAMPLE_RATE = 16000
N_MFCC = 20
DEFAULT_MODEL_PATH = Path("checkpoints/multimodal_model.joblib")


def extract_audio_features(audio_path: str | Path) -> np.ndarray:
    """
    Extract the same 40 MFCC features used during training.
    """
    path = Path(audio_path)

    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")

    # Load audio as mono at 16 kHz
    signal, sr = librosa.load(
        str(path),
        sr=SAMPLE_RATE,
        mono=True
    )

    if len(signal) == 0:
        raise ValueError(f"Audio recording is empty: {path}")

    # Extract 20 MFCC coefficients across time
    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=sr,
        n_mfcc=N_MFCC
    )

    # Calculate mean and standard deviation for each MFCC
    mean = mfcc.mean(axis=1)
    std = mfcc.std(axis=1)

    # Combine 20 means + 20 standard deviations
    features = np.concatenate([mean, std])

    if features.shape != (40,) or not np.all(np.isfinite(features)):
        raise ValueError(f"Invalid audio features: {path}")

    return features.astype(np.float32)


class EmotionPredictor:
    """
    End-to-end multimodal emotion prediction.
    """

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH
    ) -> None:

        path = Path(model_path)

        if not path.is_file():
            raise FileNotFoundError(
                f"Multimodal model not found: {path}"
            )

        # Only load checkpoint files from trusted sources
        bundle = joblib.load(path)

        required = {
            "tfidf",
            "audio_scaler",
            "classifier"
        }

        if not isinstance(bundle, dict) or not required.issubset(bundle):
            raise ValueError("Invalid model checkpoint structure")

        self.tfidf = bundle["tfidf"]
        self.scaler = bundle["audio_scaler"]
        self.classifier = bundle["classifier"]

    def predict(
        self,
        text: str,
        audio_path: str | Path
    ) -> dict[str, Any]:

        if not isinstance(text, str) or not text.strip():
            raise ValueError("Text must be a nonempty utterance")

        # Start inference timer
        start = perf_counter()

        # Step 1: Convert text into TF-IDF features
        text_features = self.tfidf.transform([text])

        # Step 2: Convert WAV into MFCC features
        audio_features = extract_audio_features(
            audio_path
        ).reshape(1, -1)

        # Step 3: Standardize audio features
        scaled_audio = self.scaler.transform(
            audio_features
        )

        # Step 4: Combine text and audio
        combined = hstack([
            text_features,
            csr_matrix(scaled_audio)
        ]).tocsr()

        # Step 5: Predict emotion
        emotion = str(
            self.classifier.predict(combined)[0]
        )

        # Step 6: Obtain probabilities
        probabilities = self.classifier.predict_proba(
            combined
        )[0]

        classes = [
            str(c) for c in self.classifier.classes_
        ]

        scores = {
            label: round(float(prob), 4)
            for label, prob in zip(classes, probabilities)
        }

        # Step 7: Generate a short robot response
        response = generate_response(text, emotion)

        # Step 8: Measure inference latency
        elapsed_ms = (perf_counter() - start) * 1000

        return {
            "emotion": emotion,
            "confidence": round(
                float(max(probabilities)), 4
            ),
            "scores": scores,
            "response": response,
            "latency_ms": round(elapsed_ms, 2)
        }