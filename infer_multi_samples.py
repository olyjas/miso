"""Single-class inference — N random val samples ranked by SI-SDRi.

Usage:
    python infer_multi_samples.py \
        --ckpt runs/miso_heavy_breathing/best-epoch=71.ckpt \
        --config configs/tse/miso_heavy_breathing.yaml \
        --data_dir data/HeavyBreathingOnly \
        --trigger heavy_breathing \
        --output_dir inference_results/

Output:
    inference_results/run_YYYYMMDD_HHMMSS_heavy_breathing/
    ├── rank1_sample04_si_sdri+12.34dB/
    │   ├── mix_input.wav
    │   ├── clean_output.wav
    │   └── gt_background.wav
    ├── rank2_sample01_si_sdri+9.81dB/
    │   ...
    └── ranking_summary.txt
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


def si_sdr(est: np.ndarray, ref: np.ndarray) -> float:
    est = est - est.mean()
    ref = ref - ref.mean()
    alpha = np.dot(est, ref) / (np.dot(ref, ref) + 1e-8)
    proj = alpha * ref
    noise = est - proj
    return 10 * np.log10(np.dot(proj, proj) / (np.dot(noise, noise) + 1e-8))


def si_sdri(output: np.ndarray, mixture_mono: np.ndarray, gt: np.ndarray) -> float:
    return si_sdr(output, gt) - si_sdr(mixture_mono, gt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--config", required=True, help="Single-class model config YAML")
    parser.add_argument("--data_dir", required=True, help="Single-class data dir (e.g. data/HeavyBreathingOnly)")
    parser.add_argument("--trigger", required=True, help="Trigger class name for labeling (e.g. heavy_breathing)")
    parser.add_argument("--n_samples", type=int, default=7)
    parser.add_argument("--output_dir", default="inference_results")
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"\nModel : {args.trigger.upper()} (single-class)")
    print(f"Ckpt  : {args.ckpt}")
    print(f"Device: {device}\n")

    model, speaker_dim = load_model(args.ckpt, device, cfg_path=args.config)

    from src.tse.train import _build_datasets
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    _, val_ds = _build_datasets(cfg, args.data_dir)
    print(f"Val dataset: {len(val_ds)} samples")

    # Pick n_samples random indices
    indices = random.sample(range(len(val_ds)), min(args.n_samples, len(val_ds)))
    label_vec = torch.ones(1)  # single-class: always [1.0]

    # Run inference and collect results
    results = []
    for i, idx in enumerate(indices):
        inputs, targets = val_ds[idx]
        mixture = inputs["mixture"].numpy()       # (2, T)
        gt = targets["target"].numpy()            # (1, T)
        gt_mono = gt[0] if gt.ndim == 2 else gt

        print(f"[Sample {i+1}/{args.n_samples}] val index {idx} ...")
        output = run_inference(model, mixture, label_vec, device, sr=args.sr)

        # RMS match to input loudness
        in_rms = np.sqrt(np.mean(mixture ** 2))
        out_rms = np.sqrt(np.mean(output ** 2))
        if out_rms > 1e-8:
            output = output * (in_rms / out_rms)

        mix_mono = mixture.mean(axis=0)
        score = si_sdri(output, mix_mono, gt_mono[:len(output)])
        print(f"  SI-SDRi: {score:+.2f} dB\n")

        results.append({
            "sample_num": i + 1,
            "val_idx": idx,
            "mixture": mixture,
            "output": output,
            "gt_mono": gt_mono,
            "si_sdri": score,
            "in_rms": in_rms,
        })

    # Sort best → worst
    results.sort(key=lambda r: r["si_sdri"], reverse=True)

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, f"run_{timestamp}_{args.trigger}")
    os.makedirs(run_dir, exist_ok=True)

    # Save files with rank in folder name
    summary_lines = [
        f"Model: {args.trigger} (single-class)",
        f"Checkpoint: {args.ckpt}",
        f"Samples: {args.n_samples}",
        f"",
        f"{'Rank':<6} {'Sample':<8} {'Val Idx':<10} {'SI-SDRi':>10} {'In RMS':>8}",
        f"{'-'*46}",
    ]

    for rank, r in enumerate(results, 1):
        folder = f"rank{rank}_sample{r['sample_num']:02d}_si_sdri{r['si_sdri']:+.2f}dB"
        out_dir = os.path.join(run_dir, folder)
        os.makedirs(out_dir, exist_ok=True)

        sf.write(os.path.join(out_dir, "mix_input.wav"), r["mixture"].T, args.sr)
        sf.write(os.path.join(out_dir, "clean_output.wav"), r["output"], args.sr)
        sf.write(os.path.join(out_dir, "gt_background.wav"), r["gt_mono"], args.sr)

        summary_lines.append(
            f"{rank:<6} {r['sample_num']:<8} {r['val_idx']:<10} {r['si_sdri']:>+10.2f} {r['in_rms']:>8.4f}"
        )

    summary_lines.append(f"\nOutputs saved to: {run_dir}/")
    summary = "\n".join(summary_lines)

    with open(os.path.join(run_dir, "ranking_summary.txt"), "w") as f:
        f.write(summary)

    print(f"\n{'='*46}")
    print(summary)
    print(f"{'='*46}")


if __name__ == "__main__":
    main()
