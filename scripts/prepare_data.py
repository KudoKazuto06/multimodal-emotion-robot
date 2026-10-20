from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

for split in ["train", "dev", "test"]:
    df = pd.read_csv(RAW_DIR / f"{split}_sent_emo.csv")

    df["audio_path"] = df.apply(
        lambda row: str(
            PROCESSED_DIR
            / "audio"
            / split
            / f"dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}.wav"
        ),
        axis=1,
    )

    df["audio_exists"] = df["audio_path"].apply(
        lambda path: Path(path).is_file()
    )

    missing = df.loc[~df["audio_exists"]]
    if not missing.empty:
        print(f"\nMissing {split} audio:")
        print(missing[["Dialogue_ID", "Utterance_ID"]])

    clean_df = df.loc[df["audio_exists"]].copy()

    output = PROCESSED_DIR / f"{split}_manifest.csv"
    clean_df.to_csv(output, index=False)

    print(f"{split}: {len(clean_df)} valid samples -> {output}")