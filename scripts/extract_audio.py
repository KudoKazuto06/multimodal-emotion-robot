from pathlib import Path
import subprocess
import argparse

def extract_audio(split):
    video_dir = Path(f"data/raw/videos/{split}")
    audio_dir = Path(f"data/processed/audio/{split}")

    audio_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted(
        video
        for video in video_dir.rglob("*.mp4")
        if not video.name.startswith("._")
    )
    print(f"Found {len(videos)} videos in {split}")

    success = 0
    failed = []

    for i, video in enumerate(videos, start=1):
        output = audio_dir / f"{video.stem}.wav"

        if output.exists() and output.stat().st_size > 44:
            success += 1
            continue

        result = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel", "error",
                "-y",
                "-i", str(video),
                "-vn",
                "-ac", "1",
                "-ar", "16000",
                "-c:a", "pcm_s16le",
                str(output)
            ],
            capture_output=True,
            text=True
        )

        if result.returncode == 0 and output.exists() and output.stat().st_size > 44:
            success += 1
        else:
            failed.append((video.name, result.stderr))
            output.unlink(missing_ok=True)

        if i % 250 == 0:
            print(f"Processed {i}/{len(videos)} | Successful: {success}")

    print(f"\nSplit: {split}")
    print(f"Successful: {success}")
    print(f"Failed: {len(failed)}")

    for name, error in failed[:10]:
        print(f"{name}: {error}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        choices=["train", "dev", "test"],
        required=True
    )
    args = parser.parse_args()

    extract_audio(args.split)