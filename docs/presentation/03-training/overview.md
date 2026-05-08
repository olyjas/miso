# 03-1. Training — Overview

This document walks through how **TFGridNet Large** training flows from the entry point down to the backend.

Details:
- [03-2. Trainer Structure](./trainer-structure.md)
- [03-3. Model Structure](./model-structure.md)
- [03-4. wandb Usage Examples](./wandb-accelerator.md)

---

## Flow from Entry Point to fit()

```
$ python -m src.tse.train \
      --config configs/tse/orange_pi.yaml \
      --data_dir /path/to/data
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│ src/tse/train.py:main()  (line 207)                             │
│                                                                 │
│  1. cfg = yaml.safe_load(args.config)                           │
│                                                                 │
│  2. _build_datasets(cfg, data_dir)                              │
│       ├─ SoundscapeDataset(split="train", fg_dir, noise_dir,    │
│       │                    hrtf_list, **common)                 │
│       └─ SoundscapeDataset(split="val", ..., val_hrtf)          │
│                                                                 │
│  3. DataLoader(train_ds, batch_size=8, num_workers=8,           │
│                pin_memory=True, drop_last=True, shuffle=True)   │
│     DataLoader(val_ds, ...) — shuffle=False                     │
│                                                                 │
│  4. _build_model(cfg) → Net(...)                                │
│       Total params: 501,738 (TFGridNet Large)                   │
│                                                                 │
│  5. optimizer = AdamW(lr=1e-3, weight_decay=0.0)                │
│     scheduler = ReduceLROnPlateau(factor=0.5, patience=5)       │
│                                                                 │
│  6. loss_fn = MultiResoFuseLoss(l1_ratio=10)                    │
│                                                                 │
│  7. trainer = create_trainer(backend="lightning")               │
│              src/trainer/__init__.py:6                          │
│                                                                 │
│  8. trainer.fit(model, train_loader, val_loader,                │
│                 loss_fn, optimizer, scheduler, cfg, metrics_fn, │
│                 resume_from=args.resume_from)                   │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│ Backend Branching                                               │
│                                                                 │
│   tc.backend == "lightning"  (default, configs/tse/*.yaml:33)   │
│   ├─→ LightningTrainerBackend.fit()                             │
│   │     src/trainer/lightning.py:94                             │
│   │     └─ pl.Trainer(accelerator="auto", devices="auto", ...)  │
│   │                                                             │
│   tc.backend == "fabric"                                        │
│   └─→ FabricTrainerBackend.fit()                                │
│         src/trainer/fabric.py:139                               │
│         └─ Fabric(accelerator="auto", devices="auto",           │
│                    precision="32-true", ...)                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Lightning vs Fabric Backend Differences

| Item | Lightning | Fabric |
|---|---|---|
| Class | `LightningTrainerBackend` | `FabricTrainerBackend` |
| Code | `src/trainer/lightning.py:84` | `src/trainer/fabric.py:125` |
| Training loop | Automatic via `pl.Trainer.fit()` | Manual loop via `_train_one_epoch()` |
| Callbacks | `ModelCheckpoint`, `LearningRateMonitor`, `EarlyStopping` automatic | Manually implemented (early stopping via `patience_counter`) |
| Logger | `pl_logger = config.get("logger", True)` (TensorBoard by default) | No separate logger; uses `logging` module |
| Mixed Precision | Configurable via `pl.Trainer(precision=...)` | Defaults to `Fabric(precision="32-true")` (configurable) |
| DDP/FSDP | Automatic via `accelerator="auto", devices="auto"` | Automatic via `accelerator="auto", devices="auto"` |
| Validation | `validation_step` invoked automatically | `_validate_one_epoch()` called manually |

> **The default backend for this model (TFGridNet Large) is `lightning`** (`configs/tse/orange_pi.yaml:43`).

---

## Core Training Hyperparameters (TFGridNet Large)

`configs/tse/orange_pi.yaml`:

```yaml
training:
  backend: "lightning"
  max_epochs: 200
  optimizer: "AdamW"
  lr: 0.001
  weight_decay: 0.0
  grad_clip: 1.0
  scheduler: "ReduceLROnPlateau"
  scheduler_params:
    factor: 0.5
    patience: 5

loss:
  l1_ratio: 10              # MultiResoFuseLoss = MR-STFT + 10 * L1

evaluation:
  metrics: ["si_sdri", "snri"]

checkpointing:
  monitor: "val/loss"       # ModelCheckpoint criterion
  mode: "min"
  save_dir: "runs/tse"
  patience: 20              # EarlyStopping (in practice, the EarlyStopping code path is only active when patience>0)

data:
  sr: 16000
  duration: 5
  batch_size: 8
  num_workers: 8
  samples_per_epoch: 20000
  num_fg_range: [1, 5]
  num_bg_range: [1, 3]
  num_noise_range: [1, 1]
  snr_range_fg: [5, 15]
  snr_range_bg: [0, 10]
```

---

## What One Epoch Looks Like

```
epoch start
  │
  │ DataLoader (num_workers=8) prefetches 8 samples/iteration
  │   └ SoundscapeDataset.__getitem__ x 8 in parallel processes
  │       (LUFS normalize, HRTF, augment, ...)
  │
  ▼
forward pass:
  │ inputs["mixture"] (8, 2, 80000)
  │ inputs["embedding"] = inputs["label_vector"]   ← Lightning wrapper performs
  │                                                    target/inputs conversion
  │ Net.forward(inputs)
  │   ├ STFT → (8, 4, T, 129)
  │   ├ MultiFiLMGuidedTFNet → (8, 5, T, 258)
  │   └ iSTFT → (8, 5, 80000)
  ▼
loss = MultiResoFuseLoss(est, gt)
  ▼
backward + optimizer.step()
  │ (handled automatically by Lightning)
  ▼
log("train/loss", loss, sync_dist=True)
  │
  ▼
next batch ...
  │
  │ samples_per_epoch=20000, batch_size=8 → 2,500 steps / epoch
  ▼
validation (end of epoch):
  │ _LitWrapper.validation_step over val_loader
  │ self.log("val/loss"), self.log("val/si_sdri"), self.log("val/snri")
  │ ReduceLROnPlateau scheduler.step(val/loss)
  │ ModelCheckpoint: saves best-{epoch:02d}.ckpt when best is updated
  ▼
next epoch (200 total)
```

Next: [Trainer Structure →](./trainer-structure.md)
