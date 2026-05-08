# 01-1. System Architecture — Overview

This document shows how the **TSE (Target Sound Extraction)** and **SED (Sound Event Detection)** pipelines cooperate to process binaural input.

Detailed diagrams are split into separate documents:

- [01-2. TSE pipeline detail](./tse-pipeline.md)
- [01-3. SED pipeline detail](./sed-pipeline.md)

---

## End-to-End Data Flow

```
                             ┌──────────────────────────────────────┐
                             │ BinauralCuratedDataset               │
                             │  scaper_fmt/{train,val,test}/{class}/│
                             │  bg_scaper_fmt/...                   │
                             │  noise_scaper_fmt/{train,val,test}/  │
                             │  hrtf/CIPIC/{*.sofa, *_hrtf.txt}     │
                             └─────────────────┬────────────────────┘
                                               │ on-the-fly
                                               ▼
                             ┌──────────────────────────────────────┐
                             │ SoundscapeDataset.__getitem__         │
                             │ (src/datasets/soundscape_dataset.py:421)│
                             │  - sample fg/bg/noise                │
                             │  - LUFS normalize                    │
                             │  - CIPIC HRTF convolution            │
                             └─────────────────┬────────────────────┘
                                               │
                                               ▼
                          ┌───────────────────────────────────────────┐
                          │ inputs = {                                │
                          │   "mixture":      (B, 2, 80000),          │
                          │   "label_vector": (B, 20),  ← multi-hot   │
                          │ }                                         │
                          │ targets = {                               │
                          │   "target":       (B, num_out, 80000),    │
                          │   "fg_labels":    list[str],              │
                          │ }                                         │
                          └────┬─────────────────────────┬────────────┘
                               │                         │
                               ▼                         ▼
                  ┌────────────────────┐    ┌──────────────────────────┐
                  │ SED                │    │ TSE                      │
                  │ ASTHuggingFace     │    │ Net  (TFGridNet Large)   │
                  │ src/sed/ast_hf.py  │    │ src/tse/net.py:31        │
                  │                    │    │   ├ STFT (chunk=96)      │
                  │ ┌────────────────┐ │    │   ├ MultiFiLMGuidedTFNet │
                  │ │ logits/scores  │ │    │   │   ├ Conv2d (proj)    │
                  │ │ softmax → 20cls│ │    │   │   ├ FiLM × 6 (preset │
                  │ └────────────────┘ │    │   │   │   = "all")       │
                  │         │          │    │   │   ├ GridNetBlock × 6 │
                  │         ▼          │    │   │   │   (LSTM intra +  │
                  │ ┌────────────────┐ │    │   │   │    LSTM inter)   │
                  │ │ multi-hot      │ │    │   │   └ ConvTranspose2d  │
                  │ │ label_vector   │─┼───→│   └ iSTFT + overlap-add  │
                  │ │ (B, 20)        │ │    │                          │
                  │ └────────────────┘ │    └────────────┬─────────────┘
                  └────────────────────┘                 │
                                                         ▼
                                       ┌────────────────────────────────┐
                                       │ output = {                     │
                                       │   "output":     (B, 5, 80000), │
                                       │   "next_state": dict,          │
                                       │ }                              │
                                       └────────────────────────────────┘
```

---

## Roles of the Two Pipelines

| Pipeline | Input | Output | Role |
|---|---|---|---|
| **SED** | binaural mixture | 20-class multi-hot label vector | Detects "which sounds are present right now?" |
| **TSE** | binaural mixture + label vector | per-source separated 1-channel waveform | Separates "extract only the selected sounds" |

> **Key data flow**: SED → label_vector → fed into TSE as FiLM conditioning.
> During training and evaluation the ground-truth label_vector is used, but at inference time the SED output becomes the input.

---

## Directory Layout (code-level)

```
fine_grained_soundscape_control_for_augmented_hearing/
├── configs/
│   ├── tse/                  ← TSE training configs (yaml)
│   │   ├── orange_pi.yaml    ← TFGridNet Large (reference for this document)
│   │   ├── raspberry_pi.yaml
│   │   ├── neuralaid.yaml
│   │   └── test_pipeline.yaml
│   └── sed/
│       └── ast_finetune.yaml
├── data/
│   ├── setup_data.py         ← entry point for dataset download/build
│   ├── pipeline/             ← Stage 1-3 (download/collect/prepare)
│   │   ├── download.py
│   │   ├── collect.py        ← generates per-source train/val/test.csv
│   │   ├── prepare.py        ← scaper_fmt symlinks + HRTF split
│   │   └── sources/          ← per-dataset classes (FSD50K, ESC-50, ...)
│   ├── class_map.yaml        ← 20-class definitions + AudioSet ID mapping
│   └── ontology.json         ← AudioSet ontology
├── src/
│   ├── tse/
│   │   ├── net.py                       ← STFT wrapper (Net)
│   │   ├── multiflim_guided_tfnet.py    ← FiLM-conditioned separator
│   │   ├── gridnet_block.py             ← intra/inter LSTM block
│   │   ├── film.py                      ← FiLM(x, emb) = x*a(emb)+b(emb)
│   │   ├── train.py                     ← TSE training entry point
│   │   ├── eval.py                      ← TSE evaluation entry point
│   │   └── model.py                     ← HF pretrained loader
│   ├── sed/
│   │   ├── ast_hf.py                    ← HF AST wrapper
│   │   ├── train.py                     ← SED training entry point
│   │   └── eval.py                      ← SED evaluation + threshold calibration
│   ├── datasets/
│   │   ├── MisophoniaDataset.py         ← legacy on-the-fly synthesis (for paper reproduction)
│   │   ├── soundscape_dataset.py        ← clean reimplementation, default for train.py (subclasses Dataset directly)
│   │   ├── multi_ch_simulator.py        ← CIPICSimulator (HRTF convolution)
│   │   └── motion_simulator.py
│   ├── trainer/
│   │   ├── base.py                      ← TrainerBackend ABC
│   │   ├── lightning.py                 ← PyTorch Lightning backend
│   │   └── fabric.py                    ← Lightning Fabric backend
│   └── metrics/
│       ├── tse.py                       ← SI-SDR, SNR
│       └── sed.py                       ← mAP, F1
└── docs/presentation/                   ← (this set of documents)
```

---

## Model Summary at a Glance (TFGridNet Large)

| Item | Value |
|---|---|
| Class | `MultiFiLMGuidedTFNet` (`src/tse/multiflim_guided_tfnet.py:20`) |
| Block type | `GridNetBlock` (`src/tse/gridnet_block.py:6`) |
| Parameters | **501,738 (0.502 M)** |
| latent_dim D | 32 |
| hidden_channels H | 64 |
| num_layers B | 6 |
| FiLM preset | `"all"` (FiLM inserted in front of all 6 blocks) |
| Input channels | 2 (binaural) |
| Output channels | 5 |
| STFT chunk / lookback / lookahead | 96 / 96 / 64 samples (6/6/4 ms @ 16 kHz) |
| nfft | 256 (→ 129 freq bins) |

Next: [TSE pipeline detail →](./tse-pipeline.md)
