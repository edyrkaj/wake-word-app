import argparse
import os
import time
import wave
from datetime import datetime

import sounddevice as sd


APP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_DIR = os.path.join(APP_DIR, "dataset")


def safe_label(label):
    return label.strip().lower().replace(" ", "_")


def write_wav(path, samples, sample_rate):
    with wave.open(path, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)  # int16
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(samples.tobytes())


def record_clip(duration, sample_rate):
    print("Recording...")
    samples = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="int16",
    )
    sd.wait()
    print("Done.")
    return samples


def main():
    parser = argparse.ArgumentParser(description="Record wake-word WAV files into dataset/")
    parser.add_argument("--label", default="hey_boss", help="Dataset label folder, e.g. hey_boss or negative")
    parser.add_argument("--duration", type=float, default=2.0, help="Seconds per recording")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Audio sample rate")
    parser.add_argument("--count", type=int, default=1, help="Number of clips to record")
    parser.add_argument("--pause", type=float, default=1.0, help="Seconds to wait between clips")
    args = parser.parse_args()

    label = safe_label(args.label)
    output_dir = os.path.join(DATASET_DIR, label)
    os.makedirs(output_dir, exist_ok=True)

    for index in range(args.count):
        if args.count > 1:
            print(f"\nClip {index + 1}/{args.count}. Get ready...")
            time.sleep(args.pause)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_path = os.path.join(output_dir, f"{label}_{timestamp}.wav")
        samples = record_clip(args.duration, args.sample_rate)
        write_wav(output_path, samples, args.sample_rate)
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
