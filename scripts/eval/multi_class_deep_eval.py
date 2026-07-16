"""Per-class evaluation for the multiclass model.

Usage:
    python scripts/eval/multi_class_deep_eval.py \
        --ckpt runs/miso/best-epoch=147.ckpt \
        --config configs/tse/miso.yaml \
        --data_dir data/MisoDataset \
        --n_per_class 50 \
        --output_dir eval_results/
"""

import argparse
import os
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch
import yaml

from infer import load_model, run_inference, TRIGGER_CLASSES


def si_sdr(est, ref):
    est = est - est.mean()
    ref = ref - ref.mean()
    alpha = np.dot(est, ref) / (np.dot(ref, ref) + 1e-8)
    proj = alpha * ref
    noise = est - proj
    return 10 * np.log10(np.dot(proj, proj) / (np.dot(noise, noise) + 1e-8))


def snr(est, ref):
    noise = est - ref
    return 10 * np.log10(np.dot(ref, ref) / (np.dot(noise, noise) + 1e-8))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--config", default="configs/tse/miso.yaml")
    parser.add_argument("--data_dir", default="data/MisoDataset")
    parser.add_argument("--n_per_class", type=int, default=50,
                        help="Number of val samples to evaluate per class")
    parser.add_argument("--output_dir", default="eval_results")
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"Device: {device}")
    print(f"Evaluating {args.n_per_class} samples per class ({args.n_per_class * 7} total)\n")

    model, speaker_dim = load_model(args.ckpt, device, cfg_path=args.config)

    from src.tse.train import _build_datasets
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    _, val_ds = _build_datasets(cfg, args.data_dir)
    print(f"Val dataset: {len(val_ds)} samples\n")

    # Collect n_per_class samples per trigger class
    class_samples = defaultdict(list)
    needed = {i: args.n_per_class for i in range(len(TRIGGER_CLASSES))}

    print("Collecting samples per class...")
    for idx in range(len(val_ds)):
        if not any(v > 0 for v in needed.values()):
            break
        inputs, targets = val_ds[idx]
        emb = inputs["embedding"]
        active = [i for i, v in enumerate(emb) if v > 0]
        if len(active) != 1:
            continue
        class_idx = active[0]
        if needed.get(class_idx, 0) > 0:
            class_samples[class_idx].append((inputs, targets))
            needed[class_idx] -= 1

    # Run inference and compute metrics per class
    results = defaultdict(lambda: {"si_sdri": [], "snri": []})

    for class_idx, samples in sorted(class_samples.items()):
        trigger = TRIGGER_CLASSES[class_idx]
        label_vec = torch.zeros(speaker_dim)
        label_vec[class_idx] = 1.0

        print(f"[{trigger}] running {len(samples)} samples...")

        for inputs, targets in samples:
            mixture = inputs["mixture"].numpy()   # (2, T)
            gt = targets["target"].numpy()
            gt_mono = gt[0] if gt.ndim == 2 else gt
            mix_mono = mixture.mean(axis=0)

            output = run_inference(model, mixture, label_vec, device, sr=args.sr)

            # RMS match to input loudness
            in_rms = np.sqrt(np.mean(mixture ** 2))
            out_rms = np.sqrt(np.mean(output ** 2))
            if out_rms > 1e-8:
                output = output * (in_rms / out_rms)

            T = min(len(output), len(gt_mono), len(mix_mono))
            si_sdri_val = si_sdr(output[:T], gt_mono[:T]) - si_sdr(mix_mono[:T], gt_mono[:T])
            snri_val = snr(output[:T], gt_mono[:T]) - snr(mix_mono[:T], gt_mono[:T])

            results[trigger]["si_sdri"].append(si_sdri_val)
            results[trigger]["snri"].append(snri_val)

    # Print and save summary table
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    header = f"\n{'─'*58}\n{'Trigger':<18} {'Samples':>8} {'SI-SDRi':>10} {'SNRi':>10}\n{'─'*58}"
    rows = []
    all_sisdri, all_snri = [], []

    for trigger in TRIGGER_CLASSES:
        if trigger not in results:
            rows.append(f"{trigger:<18} {'N/A':>8} {'N/A':>10} {'N/A':>10}")
            continue
        si = results[trigger]["si_sdri"]
        sn = results[trigger]["snri"]
        all_sisdri.extend(si)
        all_snri.extend(sn)
        rows.append(
            f"{trigger:<18} {len(si):>8} {np.mean(si):>+10.2f} {np.mean(sn):>+10.2f}"
        )

    footer = (
        f"{'─'*58}\n"
        f"{'OVERALL':<18} {len(all_sisdri):>8} "
        f"{np.mean(all_sisdri):>+10.2f} {np.mean(all_snri):>+10.2f}\n"
        f"{'─'*58}"
    )

    summary = "\n".join([header] + rows + [footer])
    print(summary)

    out_path = os.path.join(args.output_dir, f"per_class_eval_{timestamp}.txt")
    with open(out_path, "w") as f:
        f.write(f"Checkpoint: {args.ckpt}\n")
        f.write(f"Samples per class: {args.n_per_class}\n")
        f.write(summary + "\n")
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
