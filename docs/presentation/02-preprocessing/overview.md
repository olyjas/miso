# 02-1. Preprocessing — Overview

The preprocessing pipeline in this repository consists of **two stages**:

| Stage | Timing | Output | Details |
|---|---|---|---|
| **A. Offline split** | Once before training begins | `BinauralCuratedDataset/scaper_fmt/{train,val,test}/{class}/` symlinks plus per-dataset `{train,val,test}.csv` | [02-2 →](./dataset-split.md) |
| **B. Online batch** | Per sample on every epoch | `(mixture, target, label_vector)` tensors | [02-3 →](./batch-creation.md) |

---

## Stage A vs Stage B — What Happens Where

```
┌──────────────────────────────────────────────────────────────────────────┐
│ A. Offline (run once)                                                    │
│   bash scripts/setup_dataset.sh --output_dir <path>                      │
│   ├─ Stage 1: download   FSD50K, ESC-50, musdb18, DISCO, TAU, CIPIC       │
│   ├─ Stage 2: collect    each dataset → train/val/test.csv               │
│   └─ Stage 3: prepare    scaper_fmt/{class}/ symlinks, HRTF split,       │
│                          start_times.csv (silence trim metadata)         │
│                                                                          │
│   Output:                                                                │
│   <path>/BinauralCuratedDataset/                                         │
│     ├ scaper_fmt/{train,val,test}/{20-classes}/         ← FG sources     │
│     ├ bg_scaper_fmt/{train,val,test}/{class}/           ← BG sources     │
│     ├ noise_scaper_fmt/{train,val,test}/{scene}/        ← TAU urban noise│
│     ├ hrtf/CIPIC/{train,val,test}_hrtf.txt + *.sofa     ← HRTF split     │
│     └ start_times.csv                                                    │
└──────────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ B. Online (per sample during training)                                   │
│   SoundscapeDataset.__getitem__(idx)                                     │
│   src/datasets/soundscape_dataset.py:421                                 │
│                                                                          │
│   1. RNG seed = idx + np.random.randint(1e6)  (training mode)            │
│   2. n_fg = U[1,5], n_bg = U[1,3], n_noise = 1                           │
│   3. for each fg/bg:                                                     │
│        - random class → random file → random 5s snippet                  │
│        - 🟡 power filter: only accept snippets with pwr_dB > -40 dB      │
│             (_get_random_snippet, line 301) → reject silent segments     │
│             (relax by -1 dB after retries fail)                          │
│        - resample to 16kHz                                               │
│        - LUFS normalize (target = -50 + SNR_fg/bg)                       │
│      for noise:                                                          │
│        - require_power=False (urban noise kept as-is, line 369)          │
│   4. hrtf_simulator.simulate(sources, noise, seed)                       │
│        → binaural rendering with HRTF convolution                        │
│   5. mixture = sum(bi_srcs) + bi_noise                                   │
│   6. peak normalize                                                      │
│   7. AudioAugmentations.apply_random_augmentations  (training only)      │
│                                                                          │
│   returns:                                                               │
│   inputs  = {"mixture": (2, 80000), "label_vector": (20,)}               │
│   targets = {"target": (num_out, 80000), "fg_labels": [...], ...}        │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Dataset Licenses and Sources

| Dataset | Role | License | Location |
|---|---|---|---|
| FSD50K | FG (multiple classes) | Mixed CC | `raw/FSD50K/` |
| ESC-50 | FG/BG | CC-BY-NC 3.0 | `raw/ESC-50/` |
| musdb18 | FG (`singing`, `music`) | Academic only | `raw/musdb18/` |
| DISCO | BG noise | CC-BY 4.0 | `raw/disco_noises/` |
| TAU-2019 | Noise (urban scene) | NC | `raw/TAU-acoustic-sounds/` |
| CIPIC HRTF | binaural simulator | Public Domain | `raw/CIPIC-HRTF/` or `cipic-hrtf-database/` |

Because of license constraints, **redistribution as a single tar archive is not allowed**. Instead, this repository provides automated downloads together with prebuilt metadata (`data/prebuilt/metadata.tar.gz`).

---

## Time Estimates (for reference)

| Stage | Disk | Time (assuming 100 Mbps network) |
|---|---|---|
| Stage 1 download (full) | ~125 GB | 2-4 hours |
| Stage 2 collect | < 1 MB CSV | a few minutes |
| Stage 3 prepare (symlinks + silence trim) | symlinks only | 30-60 min (several hours over NFS) |
| Mini dataset (for verification) | ~250 MB | 5 min (`scripts/eval/eval_mini.sh`) |

Next: [A. Offline split details →](./dataset-split.md)
