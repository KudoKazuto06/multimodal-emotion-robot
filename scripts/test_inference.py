"""Test the end-to-end multimodal inference pipeline."""

import argparse
import json
from pathlib import Path

import pandas as pd

from src.inference import EmotionPredictor


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--text",
        type=str,
        help="Utterance text"
    )

    parser.add_argument(
        "--audio",
        type=Path,
        help="Path to a WAV audio file"
    )

    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Development dataset row to test"
    )

    args = parser.parse_args()

    # Both custom inputs must be provided together
    if (args.text is None) != (args.audio is None):
        parser.error(
            "Provide both --text and --audio, or neither"
        )

    # Automatically select an example from MELD
    if args.text is None:

        manifest = pd.read_csv(
            "data/processed/dev_manifest.csv"
        )

        row = manifest.iloc[args.index]

        text = str(row["Utterance"])
        audio = Path(str(row["audio_path"]))

        print(
            f"Ground-truth emotion: {row['Emotion']}"
        )

    else:
        text = args.text
        audio = args.audio

    print(f"Text: {text}")
    print(f"Audio: {audio}")

    # Load the trained multimodal model
    predictor = EmotionPredictor()

    # Run inference
    result = predictor.predict(text, audio)

    # Display the results
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()