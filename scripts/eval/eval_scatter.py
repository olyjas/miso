"""Per-sample SI-SDRi vs SI-SNRi scatter analysis on the validation set.

Runs inference on N val samples, computes SI-SDRi and SI-SNRi per sample,
saves a scatter plot (SI-SNRi x-axis, SI-SDRi y-axis), prints the most
divergent cases (high SI-SDRi but low SI-SNRi), saves audio clips for them,
and checks whether the mixture SNR baseline was already high for those cases.

Usage:
    python eval_scatter.py \
        --config     configs/tse/miso_tapping.yaml \
        --data_dir   /gscratch/intelligentsystems/jaszhang/data/TappingOnly \
        --checkpoint runs/miso_tapping/best-epoch=69.ckpt \
        --trigger    tapping \
        --n_samples  200 \
        --output_dir eval_scatter/tapping
"""

import argparse
import csv
import os

import numpy as np
import soundfile as sf
import torch
import yaml

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[WARNING] matplotlib not found — CSV and clips will be saved, plots skipped.")


# ── Metric helpers ─────────────────────────────────────────────────────────────

def _si_sdr(estimate, reference):
    ref_energy = np.dot(reference, reference) + 1e-8
    s_target = (np.dot(estimate, reference) / ref_energy) * reference
    e_noise = estimate - s_target
    return 10 * np.log10((np.dot(s_target, s_target) + 1e-8) / (np.dot(e_noise, e_noise) + 1e-8))


def _snr(estimate, reference):
    noise = reference - estimate
    return 10 * np.log10((np.dot(reference, reference) + 1e-8) / (np.dot(noise, noise) + 1e-8))


def compute_metrics(mix_mono, output, gt):
    T = min(len(mix_mono), len(output), len(gt))
    m, o, g = mix_mono[:T], output[:T], gt[:T]

    si_sdri = _si_sdr(o, g) - _si_sdr(m, g)

    # SI-SNRi: SI-SDR with zero-mean signals
    m_zm = m - m.mean()
    o_zm = o - o.mean()
    g_zm = g - g.mean()
    si_snri = _si_sdr(o_zm, g_zm) - _si_sdr(m_zm, g_zm)

    snri    = _snr(o, g) - _snr(m, g)
    mix_snr = _snr(m, g)   # how high was baseline SNR before model

    return si_sdri, si_snri, snri, mix_snr


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config",      required=True)
    p.add_argument("--data_dir",    required=True)
    p.add_argument("--checkpoint",  required=True)
    p.add_argument("--trigger",     required=True)
    p.add_argument("--n_samples",   type=int, default=200)
    p.add_argument("--save_clips",  type=int, default=5,
                   help="Save N most divergent clips for listening")
    p.add_argument("--output_dir",  default="eval_scatter")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    from src.tse.train import _build_datasets
    _, val_ds = _build_datasets(cfg, args.data_dir)
    sr = cfg["data"].get("sr", 16000)
    n = min(args.n_samples, len(val_ds))
    print(f"Val set: {len(val_ds)} samples  |  evaluating {n}")

    from infer import load_model, make_label_vector, run_inference as _run
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model, speaker_dim = load_model(args.checkpoint, device, cfg_path=args.config)
    label_vec = make_label_vector([args.trigger], speaker_dim=speaker_dim)

    rows = []
    for idx in range(n):
        inputs, targets = val_ds[idx]
        mixture_np = inputs["mixture"].numpy()   # (2, T)
        gt_np = targets["target"].numpy()
        if gt_np.ndim == 2:
            gt_np = gt_np[0]

        output   = _run(model, mixture_np, label_vec, device)
        mix_mono = mixture_np.mean(axis=0)

        si_sdri, si_snri, snri, mix_snr = compute_metrics(mix_mono, output, gt_np)
        rows.append(dict(
            idx=idx,
            si_sdri=round(si_sdri, 4),
            si_snri=round(si_snri, 4),
            snri=round(snri, 4),
            mix_snr=round(mix_snr, 4),
            divergence=round(si_sdri - si_snri, 4),
        ))
        if (idx + 1) % 20 == 0:
            print(f"  [{idx+1:3d}/{n}]  SI-SDRi={si_sdri:+6.2f}  SI-SNRi={si_snri:+6.2f}  div={si_sdri-si_snri:+5.2f}")

    # ── CSV ──────────────────────────────────────────────────────────────────
    csv_path = os.path.join(args.output_dir, "per_sample.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved: {csv_path}")

    # ── Summary stats ────────────────────────────────────────────────────────
    si_sdri_arr = np.array([r["si_sdri"] for r in rows])
    si_snri_arr = np.array([r["si_snri"] for r in rows])
    snri_arr    = np.array([r["snri"]    for r in rows])
    mix_snr_arr = np.array([r["mix_snr"] for r in rows])

    print(f"\n=== {args.trigger}  (n={n}) ===")
    print(f"  SI-SDRi  mean={si_sdri_arr.mean():+.2f}  median={np.median(si_sdri_arr):+.2f}  std={si_sdri_arr.std():.2f}")
    print(f"  SI-SNRi  mean={si_snri_arr.mean():+.2f}  median={np.median(si_snri_arr):+.2f}  std={si_snri_arr.std():.2f}")
    print(f"  SNRi     mean={snri_arr.mean():+.2f}  median={np.median(snri_arr):+.2f}  std={snri_arr.std():.2f}")
    print(f"  mix_SNR  mean={mix_snr_arr.mean():+.2f}  (baseline mixture SNR before model)")
    print(f"  % samples with SI-SDRi > 0: {(si_sdri_arr > 0).mean()*100:.1f}%")
    print(f"  % samples with SI-SNRi > 0: {(si_snri_arr > 0).mean()*100:.1f}%")

    # ── Scatter plot: SI-SNRi (x) vs SI-SDRi (y) ─────────────────────────────
    if HAS_MPL:
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(si_snri_arr, si_sdri_arr, alpha=0.45, s=20, color="#4C72B0", zorder=3,
                   label=f"n={n} val samples")
        lo = min(si_snri_arr.min(), si_sdri_arr.min()) - 3
        hi = max(si_snri_arr.max(), si_sdri_arr.max()) + 3
        ax.plot([lo, hi], [lo, hi], "k--", lw=1.2, alpha=0.5, label="y = x")
        ax.axhline(0, color="gray", lw=0.7, alpha=0.4, linestyle=":")
        ax.axvline(0, color="gray", lw=0.7, alpha=0.4, linestyle=":")
        ax.set_xlabel("SI-SNRi (dB)", fontsize=12)
        ax.set_ylabel("SI-SDRi (dB)", fontsize=12)
        ax.set_title(f"{args.trigger} — SI-SDRi vs SI-SNRi  (n={n})", fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        scatter_path = os.path.join(args.output_dir, "scatter.png")
        fig.savefig(scatter_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {scatter_path}")

        fig2, axes = plt.subplots(1, 2, figsize=(12, 4))
        for ax2, arr, label, color in [
            (axes[0], si_sdri_arr, "SI-SDRi (dB)", "#4C72B0"),
            (axes[1], si_snri_arr, "SI-SNRi (dB)", "#DD8452"),
        ]:
            ax2.hist(arr, bins=30, color=color, alpha=0.75, edgecolor="white", linewidth=0.5)
            ax2.axvline(arr.mean(), color="red", lw=1.5, linestyle="--", label=f"mean={arr.mean():.2f}")
            ax2.axvline(0, color="black", lw=0.8, alpha=0.5)
            ax2.set_xlabel(label, fontsize=11)
            ax2.set_ylabel("count", fontsize=11)
            ax2.legend(fontsize=10)
            ax2.grid(True, alpha=0.2)
        fig2.suptitle(f"{args.trigger} — metric distributions  (n={n})", fontsize=13)
        fig2.tight_layout()
        dist_path = os.path.join(args.output_dir, "distributions.png")
        fig2.savefig(dist_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {dist_path}")
    else:
        print("(plots skipped — no matplotlib; load per_sample.csv locally to plot)")

    # ── Most divergent: high SI-SDRi, low SI-SNRi ────────────────────────────
    divergent = sorted(rows, key=lambda r: r["divergence"], reverse=True)[: args.save_clips]
    print(f"\nTop {args.save_clips} divergent samples (SI-SDRi high, SI-SNRi low):")
    print(f"  {'idx':>4}  {'SI-SDRi':>8}  {'SI-SNRi':>8}  {'diff':>7}  {'mix_SNR':>8}")
    for r in divergent:
        print(f"  {r['idx']:4d}  {r['si_sdri']:+8.2f}  {r['si_snri']:+8.2f}  "
              f"{r['divergence']:+7.2f}  {r['mix_snr']:+8.2f}")

    # Save audio clips for divergent samples
    clips_dir = os.path.join(args.output_dir, "divergent_clips")
    os.makedirs(clips_dir, exist_ok=True)
    for r in divergent:
        idx = r["idx"]
        inputs, targets = val_ds[idx]
        mixture_np = inputs["mixture"].numpy()
        gt_np = targets["target"].numpy()
        if gt_np.ndim == 2:
            gt_np = gt_np[0]
        output = _run(model, mixture_np, label_vec, device)

        tag = f"idx{idx:03d}_sisdri{r['si_sdri']:+.1f}_sisnri{r['si_snri']:+.1f}"
        cdir = os.path.join(clips_dir, tag)
        os.makedirs(cdir, exist_ok=True)
        sf.write(os.path.join(cdir, "mixture.wav"), mixture_np.T, sr)
        gt_norm = gt_np / (np.abs(gt_np).max() + 1e-8) * 0.8
        sf.write(os.path.join(cdir, "ground_truth.wav"), gt_norm, sr)
        out_norm = output / (np.abs(output).max() + 1e-8) * 0.8
        sf.write(os.path.join(cdir, "model_output.wav"), out_norm, sr)

    print(f"\nSaved {len(divergent)} clip sets → {clips_dir}/")
    print("  Each folder: mixture.wav | ground_truth.wav | model_output.wav")


if __name__ == "__main__":
    main()