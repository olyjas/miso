# 03-1. 학습 — 개요

본 문서는 **TFGridNet Large** 학습이 진입점에서 backend 까지 어떻게 흘러가는지 보여줍니다.

세부:
- [03-2. Trainer 구조](./trainer-structure.md)
- [03-3. 모델 구조](./model-structure.md)
- [03-4. wandb 사용 예제](./wandb-accelerator.md)

---

## 진입점부터 fit() 까지의 흐름

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
│ Backend 분기                                                    │
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

## Lightning vs Fabric 백엔드 차이

| 항목 | Lightning | Fabric |
|---|---|---|
| 클래스 | `LightningTrainerBackend` | `FabricTrainerBackend` |
| 코드 | `src/trainer/lightning.py:84` | `src/trainer/fabric.py:125` |
| 학습 루프 | `pl.Trainer.fit()` 자동 | `_train_one_epoch()` 수동 루프 |
| 콜백 | `ModelCheckpoint`, `LearningRateMonitor`, `EarlyStopping` 자동 | 수동 구현 (early stopping `patience_counter`) |
| Logger | `pl_logger = config.get("logger", True)` (기본 TensorBoard) | 별도 logger 없음, `logging` 모듈 |
| Mixed Precision | `pl.Trainer(precision=...)` 으로 설정 가능 | `Fabric(precision="32-true")` 기본 (config 변경 가능) |
| DDP/FSDP | `accelerator="auto", devices="auto"` 으로 자동 | `accelerator="auto", devices="auto"` 으로 자동 |
| Validation | `validation_step` 자동 호출 | `_validate_one_epoch()` 수동 호출 |

> **본 모델 (TFGridNet Large) 의 기본 백엔드는 `lightning`** (`configs/tse/orange_pi.yaml:43`).

---

## 핵심 학습 hyperparameter (TFGridNet Large)

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
  patience: 20              # EarlyStopping (실제로는 EarlyStopping 코드 path 가 patience>0 일 때만 활성)

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

## 한 epoch 가 도는 모습

```
epoch start
  │
  │ DataLoader (num_workers=8) 가 8 sample/iteration prefetch
  │   └ SoundscapeDataset.__getitem__ x 8 in parallel processes
  │       (LUFS normalize, HRTF, augment, ...)
  │
  ▼
forward pass:
  │ inputs["mixture"] (8, 2, 80000)
  │ inputs["embedding"] = inputs["label_vector"]   ← Lightning wrapper 가
  │                                                    target/inputs 변환
  │ Net.forward(inputs)
  │   ├ STFT → (8, 4, T, 129)
  │   ├ MultiFiLMGuidedTFNet → (8, 5, T, 258)
  │   └ iSTFT → (8, 5, 80000)
  ▼
loss = MultiResoFuseLoss(est, gt)
  ▼
backward + optimizer.step()
  │ (Lightning 이 자동)
  ▼
log("train/loss", loss, sync_dist=True)
  │
  ▼
다음 batch ...
  │
  │ samples_per_epoch=20000, batch_size=8 → 2,500 step / epoch
  ▼
validation (epoch 끝):
  │ val_loader 로 _LitWrapper.validation_step
  │ self.log("val/loss"), self.log("val/si_sdri"), self.log("val/snri")
  │ ReduceLROnPlateau scheduler.step(val/loss)
  │ ModelCheckpoint: best 갱신 시 best-{epoch:02d}.ckpt 저장
  ▼
다음 epoch (총 200 회)
```

다음: [Trainer 구조 →](./trainer-structure.md)
