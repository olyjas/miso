"""Batch inference — one random val sample per trigger class, with clean output.

Usage:
    python batch_infer.py \
        --ckpt runs/miso/best-epoch=48-v1.ckpt \
        --config configs/tse/miso.yaml \
        --data_dir data/MisoDataset \
        --output_dir inference_results/

Output:
    inference_results/run_YYYYMMDD_HHMMSS/
    ├── chewing/
    │   ├── mix_input.wav       ← background + chewing trigger
    │   ├── clean_output.wav    ← model output (trigger removed)
    │   └── gt_background.wav  ← perfect ground truth
    ├── cough/
    │   ├── mix_input.wav
    │   ├── clean_output.wav
    │   └── gt_background.wav
    └── ...
"""

import argparse
import os
import random
from datetime import datetime

import numpy as np
import soundfile as sf
import torch
import yaml

from infer import load_model, run_inference, TRIGGER_CLASSES, NUM_CLASSES


def find_sample_for_class(val_ds, class_idx, max_tries=300):
    """Search randomly through val_ds to find a sample with the given trigger class."""
    indices = list(range(len(val_ds)))
    random.shuffle(indices)
    for idx in indices[:max_tries]:
        inputs, targets = val_ds[idx]
        if inputs["embedding"][class_idx] > 0:
            return inputs, targets
    return None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--config", default="configs/tse/miso.yaml")
    parser.add_argument("--data_dir", default="data/MisoDataset")
    parser.add_argument("--output_dir", default="inference_results")
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--triggers", nargs="+", default=["all"],
        help="Trigger classes to test, or 'all'")
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"Device: {device}")

    # Load model
    model, speaker_dim = load_model(args.ckpt, device, cfg_path=args.config)

    # Load val dataset (same as training pipeline)
    from src.tse.train import _build_datasets
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    _, val_ds = _build_datasets(cfg, args.data_dir)
    print(f"Val dataset: {len(val_ds)} samples")

    # Resolve which classes to test
    triggers = TRIGGER_CLASSES if args.triggers == ["all"] else args.triggers

    # Create output directory
    run_dir = os.path.join(args.output_dir, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"\nOutputs → {run_dir}/\n")

    summary = []
    for trigger in triggers:
        class_idx = TRIGGER_CLASSES.index(trigger)
        print(f"[{trigger}] Searching for val sample...")

        inputs, targets = find_sample_for_class(val_ds, class_idx)
        if inputs is None:
            print(f"  WARNING: no sample found for {trigger}, skipping")
            continue

        mixture = inputs["mixture"].numpy()   # (2, T)
        gt = targets["target"].numpy()         # (1, T) or (2, T)
        label_vec = inputs["embedding"]

        print(f"  Input shape: {mixture.shape}  ({mixture.shape[1]/args.sr:.1f}s)")

        # Run inference (also prints Raw model output + Peak scale factor)
        out_dir = os.path.join(run_dir, trigger)
        os.makedirs(out_dir, exist_ok=True)

        output = run_inference(model, mixture, label_vec, device, sr=args.sr)

        # RMS-match to input loudness
        in_rms = np.sqrt(np.mean(mixture ** 2))
        out_rms = np.sqrt(np.mean(output ** 2))
        if out_rms > 1e-8:
            output = output * (in_rms / out_rms)

        print(f"  Input RMS: {in_rms:.4f}  Output RMS: {np.sqrt(np.mean(output**2)):.4f}")

        # Save all three files
        sf.write(os.path.join(out_dir, "mix_input.wav"), mixture.T, args.sr)
        sf.write(os.path.join(out_dir, "clean_output.wav"), output, args.sr)
        gt_mono = gt[0] if gt.ndim == 2 else gt
        sf.write(os.path.join(out_dir, "gt_background.wav"), gt_mono, args.sr)

        print(f"  Saved → {out_dir}/")
        summary.append((trigger, in_rms, np.sqrt(np.mean(output**2))))

    # Summary
    print(f"\n{'─'*50}")
    print(f"{'Trigger':<18} {'In RMS':>8} {'Out RMS':>8}")
    print(f"{'─'*50}")
    for trigger, in_rms, out_rms in summary:
        print(f"{trigger:<18} {in_rms:>8.4f} {out_rms:>8.4f}")
    print(f"{'─'*50}")
    print(f"\nDone. {len(summary)}/7 classes saved to: {run_dir}/")


if __name__ == "__main__":
    main()
