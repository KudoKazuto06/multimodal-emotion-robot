from pathlib import Path
import pandas as pd

for split in ["train", "dev", "test"]:
    csv_path = Path(f"data/raw/{split}_sent_emo.csv")
    audio_dir = Path(f"data/processed/audio/{split}")

    df = pd.read_csv(csv_path)

    expected = {
        f"dia{row.Dialogue_ID}_utt{row.Utterance_ID}.wav"
        for row in df.itertuples()
    }

    actual = {
        file.name
        for file in audio_dir.glob("*.wav")
    }

    matched = expected & actual
    missing = expected - actual
    extra = actual - expected

    print(f"\n=== {split.upper()} ===")
    print(f"CSV rows: {len(df)}")
    print(f"Expected unique audio files: {len(expected)}")
    print(f"Matching audio files: {len(matched)}")
    print(f"Missing audio files: {len(missing)}")
    print(f"Extra audio files: {len(extra)}")

    if missing:
        print("Missing examples:", sorted(missing)[:10])

    if extra:
        print("Extra examples:", sorted(extra)[:10])