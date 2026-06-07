"""Misophonia trigger removal — inference script.

Usage:
    python infer.py --ckpt runs/miso/best-epoch=02.ckpt \
                    --input my_recording.wav \
                    --trigger chewing \
                    --output clean_background.wav

The model takes a binaural (2-channel) or mono input and outputs the
background with the specified trigger removed.

Trigger classes (index → name):
  0: chewing
  1: cough
  2: drinking
  3: heavy_breathing
  4: sneeze
  5: sniffling
  6: tapping
"""

import argparse
import sys
import torch
import numpy as np
import soundfile as sf

# --- Class ordering must match sorted(os.listdir(fg_dir/train)) ---
TRIGGER_CLASSES = [
    "chewing",          # 0
    "cough",            # 1
    "drinking",         # 2
    "heavy_breathing",  # 3
    "sneeze",           # 4
    "sniffling",        # 5
    "tapping",          # 6
]
NUM_CLASSES = len(TRIGGER_CLASSES)


def load_model(ckpt_path: str, device: torch.device, cfg_path: str = "configs/tse/miso.yaml"):
    from src.tse.train import _build_model
    from src.trainer import normalize_state_dict
    import yaml

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    model = _build_model(cfg)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = normalize_state_dict(ckpt)
    model.load_state_dict(sd, strict=True)
    model.to(device)
    model.eval()
    print(f"Loaded checkpoint: {ckpt_path}")
    speaker_dim = cfg.get("model", {}).get("speaker_dim", NUM_CLASSES)
    return model, speaker_dim


def load_audio(path: str, sr: int = 16000) -> np.ndarray:
    """Load audio and return as (2, T) numpy float32."""
    audio, file_sr = sf.read(path, dtype="float32")

    # Resample if needed
    if file_sr != sr:
        try:
            import torchaudio
            t = torch.from_numpy(audio.T if audio.ndim > 1 else audio).float()
            if t.ndim == 1:
                t = t.unsqueeze(0)
            resampler = torchaudio.transforms.Resample(file_sr, sr)
            audio = resampler(t).numpy()
            audio = audio.T if audio.shape[0] > 1 else audio.squeeze(0)
        except ImportError:
            import librosa
            if audio.ndim > 1:
                audio = audio[:, 0]
            audio = librosa.resample(audio, orig_sr=file_sr, target_sr=sr)

    # Ensure (2, T) binaural shape
    if audio.ndim == 1:
        audio = np.stack([audio, audio], axis=0)   # mono → fake binaural
    elif audio.shape[1] == 2:
        audio = audio.T                             # (T, 2) → (2, T)
    elif audio.shape[0] == 2:
        pass                                        # already (2, T)
    else:
        # Multi-channel: take first two
        audio = audio[:, :2].T

    return audio.astype(np.float32)


def make_label_vector(triggers: list[str], speaker_dim: int = NUM_CLASSES) -> torch.Tensor:
    if speaker_dim == 1:
        # Single-class model: label is always [1.0] meaning "remove this trigger"
        return torch.ones(1)
    vec = torch.zeros(speaker_dim)
    for t in triggers:
        t = t.lower().strip()
        if t not in TRIGGER_CLASSES:
            print(f"WARNING: unknown trigger '{t}'. Valid classes: {TRIGGER_CLASSES}")
            continue
        vec[TRIGGER_CLASSES.index(t)] = 1.0
    return vec


def run_inference(
    model,
    audio_np: np.ndarray,   # (2, T)
    label_vec: torch.Tensor, # (speaker_dim,)
    device: torch.device,
    chunk_sec: float = 5.0,
    sr: int = 16000,
) -> np.ndarray:
    """Run the model on the full audio at once (matches training behaviour)."""
    T = audio_np.shape[1]
    x = torch.from_numpy(audio_np).unsqueeze(0).to(device)  # (1, 2, T)

    # Peak-normalise (same as training)
    peak = x.abs().max()
    if peak > 1e-6:
        x = x / peak

    embedding = label_vec.to(device).unsqueeze(0)  # (1, speaker_dim)
    inputs = {"mixture": x, "embedding": embedding}

    with torch.no_grad():
        out = model(inputs)["output"]   # (1, 1, T)

    out_mono = out[0, 0].cpu().numpy()  # (T,)

    print(f"Raw model output: min={out_mono.min():.6f} max={out_mono.max():.6f} std={out_mono.std():.6f}")
    print(f"Peak scale factor: {peak.item():.6f}")

    if peak.item() > 1e-6:
        out_mono = out_mono * peak.item()

    return out_mono[:T]


def main():
    parser = argparse.ArgumentParser(description="Remove misophonia triggers from audio")
    parser.add_argument("--ckpt", required=True, help="Path to .ckpt checkpoint")
    parser.add_argument("--config", default="configs/tse/miso.yaml", help="Path to model config YAML")
    parser.add_argument("--input", required=True, help="Input audio file (WAV/FLAC)")
    parser.add_argument(
        "--trigger",
        required=True,
        nargs="+",
        help=f"Trigger class(es) to remove. Choices: {TRIGGER_CLASSES}",
    )
    parser.add_argument("--output", default="output.wav", help="Output WAV file")
    parser.add_argument("--sr", type=int, default=16000, help="Sample rate (default 16000)")
    parser.add_argument("--chunk_sec", type=float, default=5.0, help="Processing chunk length in seconds")
    parser.add_argument("--cpu", action="store_true", help="Force CPU (no GPU)")
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"Device: {device}")

    model, speaker_dim = load_model(args.ckpt, device, cfg_path=args.config)

    print(f"Loading input: {args.input}")
    audio = load_audio(args.input, sr=args.sr)
    print(f"Input shape: {audio.shape}  ({audio.shape[1]/args.sr:.1f}s)")

    label_vec = make_label_vector(args.trigger, speaker_dim=speaker_dim)
    active = args.trigger if speaker_dim == 1 else [TRIGGER_CLASSES[i] for i, v in enumerate(label_vec) if v > 0]
    print(f"Removing triggers: {active}  (label vector: {label_vec.tolist()})")

    print("Running inference...")
    output = run_inference(model, audio, label_vec, device, chunk_sec=args.chunk_sec, sr=args.sr)

    # Normalize output to audible level (model output scale is arbitrary due to SI loss)
    out_peak = np.abs(output).max()
    if out_peak > 1e-8:
        output = output / out_peak * 0.8

    sf.write(args.output, output, args.sr)
    print(f"Saved output: {args.output}")

    # Quick quality check
    input_rms = np.sqrt(np.mean(audio**2))
    output_rms = np.sqrt(np.mean(output**2))
    print(f"Input RMS: {input_rms:.4f}  Output RMS: {output_rms:.4f}")
    print("Done.")


if __name__ == "__main__":
    main()
