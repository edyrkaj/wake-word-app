import argparse
import glob
import os
import random
import wave

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, Dataset

from onnxruntime.quantization import QuantType, quantize_dynamic  # pyright: ignore[reportMissingImports]


HERE = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.abspath(os.path.join(HERE, ".."))
DATASET_DIR = os.path.join(APP_DIR, "dataset")
PUBLIC_DIR = os.path.join(APP_DIR, "public")
WEIGHTS_PATH = os.path.join(HERE, "wake_word.pt")
MODEL_FP32 = os.path.join(HERE, "model.onnx")
MODEL_QUANT = os.path.join(PUBLIC_DIR, "model_tiny.onnx")

SAMPLE_RATE = 16000
N_MELS = 40
N_FRAMES = 99
N_FFT = 512
HOP_LENGTH = 160
WINDOW_SAMPLES = SAMPLE_RATE

# Default folder name under dataset/ for class 1; override with --positive-label.
DEFAULT_POSITIVE_LABEL = "hey_boss"


class WakeWordModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, groups=16),
            nn.Conv2d(16, 32, kernel_size=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(32, 2)

    def logits(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

    def forward(self, x):
        # Softmax is applied here so the exported ONNX matches what the JS app
        # already expects: outputs.output.data[1] is the wake-word probability.
        return torch.softmax(self.logits(x), dim=1)


def load_wav(path):
    """Read a PCM WAV file with stdlib `wave`, return (tensor[1, N], sample_rate)."""
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        frames = wf.readframes(wf.getnframes())

    if sampwidth == 2:
        data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        data = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
    elif sampwidth == 1:
        data = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sampwidth} bytes ({path})")

    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return torch.from_numpy(data).unsqueeze(0), sr


def load_audio_to_window(path):
    wav, sr = load_wav(path)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    n = wav.shape[1]
    if n < WINDOW_SAMPLES:
        wav = F.pad(wav, (0, WINDOW_SAMPLES - n))
    elif n > WINDOW_SAMPLES:
        start = (n - WINDOW_SAMPLES) // 2
        wav = wav[:, start : start + WINDOW_SAMPLES]
    return wav


def wav_to_mel(wav, mel_transform):
    mel = mel_transform(wav)
    mel = torch.log(mel + 1e-6)
    T = mel.shape[-1]
    if T < N_FRAMES:
        mel = F.pad(mel, (0, N_FRAMES - T))
    elif T > N_FRAMES:
        mel = mel[..., :N_FRAMES]
    return mel  # [1, N_MELS, N_FRAMES]


class WakeWordDataset(Dataset):
    def __init__(self, items):
        self.items = items
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            mel_scale="htk",
            norm=None,
        )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        wav = load_audio_to_window(path)
        mel = wav_to_mel(wav, self.mel)
        return mel, label


def collect_files(positive_label):
    if not os.path.isdir(DATASET_DIR):
        raise FileNotFoundError(f"Dataset folder not found: {DATASET_DIR}")
    items = []
    for entry in sorted(os.listdir(DATASET_DIR)):
        folder = os.path.join(DATASET_DIR, entry)
        if not os.path.isdir(folder):
            continue
        label = 1 if entry == positive_label else 0
        for wav_path in sorted(glob.glob(os.path.join(folder, "*.wav"))):
            items.append((wav_path, label))
    return items


def class_weights_tensor(positives, negatives, balance_weights, device):
    """Inverse-frequency weights so rare positives get higher loss when balance_weights."""
    if not balance_weights:
        return None
    n = positives + negatives
    # CrossEntropyLoss weight[i] = n / (num_classes * count_i)
    w0 = n / (2.0 * max(negatives, 1))
    w1 = n / (2.0 * max(positives, 1))
    return torch.tensor([w0, w1], dtype=torch.float32, device=device)


def train(epochs, batch_size, lr, val_split, seed, positive_label, balance_weights):
    items = collect_files(positive_label)
    if not items:
        raise RuntimeError(f"No .wav files found under {DATASET_DIR}")

    positives = sum(1 for _, l in items if l == 1)
    negatives = len(items) - positives
    print(f"Dataset: {len(items)} clips ({positives} positive, {negatives} negative)")
    if positives == 0 or negatives == 0:
        raise RuntimeError(
            f"Need both positive ({positive_label}/) and negative (any other folder) clips."
        )

    random.seed(seed)
    torch.manual_seed(seed)
    random.shuffle(items)
    val_count = max(1, int(len(items) * val_split))
    val_items = items[:val_count]
    train_items = items[val_count:]

    train_loader = DataLoader(WakeWordDataset(train_items), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(WakeWordDataset(val_items), batch_size=batch_size)

    model = WakeWordModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    device = next(model.parameters()).device
    cw = class_weights_tensor(positives, negatives, balance_weights, device)
    if cw is not None:
        print(f"CrossEntropy class weights (neg, pos): {cw.tolist()}")
    criterion = nn.CrossEntropyLoss(weight=cw)

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for mel, label in train_loader:
            optimizer.zero_grad()
            logits = model.logits(mel)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()
            running += loss.item() * mel.shape[0]
        train_loss = running / len(train_loader.dataset)

        model.eval()
        correct = 0
        with torch.no_grad():
            for mel, label in val_loader:
                pred = model.logits(mel).argmax(dim=1)
                correct += (pred == label).sum().item()
        val_acc = correct / len(val_loader.dataset) if len(val_loader.dataset) else 0.0

        print(f"Epoch {epoch:3d} | train_loss={train_loss:.4f} | val_acc={val_acc:.3f}")

    return model


def report_positive_wake_scores(model, items):
    """Mean/min/max softmax(class=1) on positive training files — compare to browser threshold."""
    pos_paths = [p for p, lab in items if lab == 1]
    if not pos_paths:
        print("No positive clips to score.")
        return
    pos_items = [(p, 1) for p in pos_paths]
    loader = DataLoader(WakeWordDataset(pos_items), batch_size=16, shuffle=False)
    model.eval()
    probs = []
    with torch.no_grad():
        for mel, _ in loader:
            p1 = model(mel)[:, 1].cpu().numpy()
            probs.append(p1)
    x = np.concatenate(probs)
    print(
        f"Wake softmax on {len(x)} positive clips — min={x.min():.3f} "
        f"mean={x.mean():.3f} max={x.max():.3f} (target >0.5 in browser after quant)"
    )


def export_onnx(model):
    os.makedirs(PUBLIC_DIR, exist_ok=True)
    model.eval()
    dummy = torch.randn(1, 1, N_MELS, N_FRAMES)
    torch.onnx.export(
        model,
        dummy,
        MODEL_FP32,
        input_names=["input"],
        output_names=["output"],
        dynamo=False,
        opset_version=17,
    )
    quantize_dynamic(MODEL_FP32, MODEL_QUANT, weight_type=QuantType.QUInt8)
    print(f"Exported quantized model: {MODEL_QUANT}")


def main():
    parser = argparse.ArgumentParser(description="Train wake-word model from dataset/")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-export", action="store_true", help="Skip ONNX export")
    parser.add_argument(
        "--positive-label",
        default=DEFAULT_POSITIVE_LABEL,
        help="dataset/<name>/ folder treated as wake word (class 1). Must match record_dataset --label.",
    )
    parser.add_argument(
        "--no-balance-weights",
        action="store_true",
        help="Disable inverse-frequency class weights (use plain CE).",
    )
    args = parser.parse_args()

    model = train(
        args.epochs,
        args.batch_size,
        args.lr,
        args.val_split,
        args.seed,
        args.positive_label,
        balance_weights=not args.no_balance_weights,
    )
    report_positive_wake_scores(model, collect_files(args.positive_label))
    torch.save(model.state_dict(), WEIGHTS_PATH)
    print(f"Saved weights: {WEIGHTS_PATH}")
    if not args.no_export:
        export_onnx(model)


if __name__ == "__main__":
    main()
