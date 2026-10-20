from pathlib import Path
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm

PROCESSED_DIR = Path("data/processed")
FEATURE_DIR = PROCESSED_DIR / "features"
FEATURE_DIR.mkdir(parents=True, exist_ok=True)

N_MFCC = 20
SAMPLE_RATE = 16000


def extract_features(audio_path):
    # Load audio waveform at 16 kHz
    signal, sr = librosa.load(
        audio_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    if len(signal) == 0:
        raise ValueError("Empty audio recording")

    # Extract Mel-frequency cepstral coefficients
    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=sr,
        n_mfcc=N_MFCC
    )

    # Summarize features across time
    mean = np.mean(mfcc, axis=1)
    std = np.std(mfcc, axis=1)

    # Combine into one 40-dimensional vector
    features = np.concatenate([mean, std])

    return features


def process_split(split):
    df = pd.read_csv(
        PROCESSED_DIR / f"{split}_manifest.csv"
    )

    features = []
    valid_indices = []
    failed = []

    for index, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc=f"Processing {split}"
    ):
        try:
            audio_path = Path(row["audio_path"])
            vector = extract_features(audio_path)

            if not np.all(np.isfinite(vector)):
                raise ValueError("Non-finite audio features")

            features.append(vector)
            valid_indices.append(index)

        except Exception as e:
            failed.append((index, str(e)))

    X = np.asarray(features, dtype=np.float32)
    clean_df = df.iloc[valid_indices].reset_index(drop=True)

    np.save(FEATURE_DIR / f"{split}_audio.npy", X)
    clean_df.to_csv(
        FEATURE_DIR / f"{split}_aligned.csv",
        index=False
    )

    print(f"\n{split.upper()}")
    print("Features shape:", X.shape)
    print("Failed:", len(failed))

    for index, error in failed[:10]:
        print(f"Row {index}: {error}")


if __name__ == "__main__":
    for split in ["train", "dev", "test"]:
        process_split(split)