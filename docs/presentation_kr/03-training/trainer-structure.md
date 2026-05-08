# 03-2. Trainer 구조

[← 학습 개요](./overview.md)

---

## 1. 인터페이스 — `TrainerBackend` ABC

`src/trainer/base.py:11`

```
┌─────────────────────────────────────────────────────────────┐
│ class TrainerBackend(ABC):                                  │
│                                                             │
│   @abstractmethod                                           │
│   def fit(model, train_loader, val_loader,                  │
│           loss_fn, optimizer, scheduler,                    │
│           config, metrics_fn) -> dict                       │
│                                                             │
│   @abstractmethod                                           │
│   def validate(model, val_loader, loss_fn,                  │
│                config, metrics_fn) -> dict                  │
│                                                             │
│   @abstractmethod                                           │
│   def save_checkpoint(model, optimizer, epoch,              │
│                       metrics, path) -> None                │
│                                                             │
│   @abstractmethod                                           │
│   def load_checkpoint(path, model, optimizer) -> dict       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

이 인터페이스 덕에 모델/데이터셋 코드는 backend 와 독립적으로 작성됩니다.

### Factory 함수

`src/trainer/__init__.py:6`

```python
def create_trainer(backend: str = "lightning", **kwargs) -> TrainerBackend:
    if backend == "lightning":
        from .lightning import LightningTrainerBackend
        return LightningTrainerBackend(**kwargs)
    elif backend == "fabric":
        from .fabric import FabricTrainerBackend
        return FabricTrainerBackend(**kwargs)
    raise ValueError(f"Unknown backend: {backend}")
```

---

## 2. Lightning 백엔드 — `LightningTrainerBackend`

`src/trainer/lightning.py:89`

### 2-1. 구조

```
LightningTrainerBackend
  │
  ├─ _LitWrapper (pl.LightningModule)             # src/trainer/lightning.py:23
  │     ├ training_step:   loss = loss_fn(est, gt).mean()
  │     │                  self.log("train/loss", ...)
  │     ├ validation_step: 동일 + self.log("val/loss", ...)
  │     │                  + self.log("val/si_sdri"), ("val/snri")
  │     └ configure_optimizers: {"optimizer": ..., "lr_scheduler": {...}}
  │
  └─ fit():
        callbacks = [ModelCheckpoint(monitor="val/loss", mode="min", ...),
                     LearningRateMonitor(),
                     # EarlyStopping(patience=es_cfg.patience) if patience>0 ]
        trainer = pl.Trainer(
            max_epochs=200,
            accelerator="auto",          # GPU/CPU 자동 선택
            devices="auto",              # 사용 가능한 모든 device
            callbacks=callbacks,
            logger=pl_logger,            # config["logger"] 또는 True
            **self._extra_kwargs,        # 사용자 정의 옵션
        )
        trainer.fit(lit_model, train_loader, val_loader)
```

### 2-2. `_LitWrapper.training_step`

`src/trainer/lightning.py:50`

```python
def training_step(self, batch, batch_idx):
    inputs, targets = batch
    outputs = self.model(inputs)
    loss = self.loss_fn(outputs, targets, inputs)
    self.log("train/loss", loss, prog_bar=True, sync_dist=True)
    return loss
```

> **시그니처 통일 (commit `7a3ad68`)**: trainer 는 dict 추출을 안 하고 `(outputs, targets, inputs)` 3-arg 로 호출. dict→tensor 추출은 task 별 adapter (`src/tse/train.py:_build_loss_fn` / `src/sed/train.py:_build_loss_fn`) 가 책임.

### 2-3. `_LitWrapper.validation_step`

`src/trainer/lightning.py:60`

```python
def validation_step(self, batch, batch_idx):
    inputs, targets = batch
    outputs = self.model(inputs)
    loss = self.loss_fn(outputs, targets, inputs)
    self.log("val/loss", loss, prog_bar=True, sync_dist=True)

    if self.metrics_fn is not None:
        metrics = self.metrics_fn(outputs, targets, inputs)
        for key, value in metrics.items():
            self.log(f"val/{key}", value, prog_bar=True, sync_dist=True)
```

`metrics_fn` 은 `src/tse/train.py:_build_metrics_fn` (또는 SED 의 `src/sed/train.py:_build_metrics_fn`) 에서 만들어진 adapter.

> **TSE metric — 패딩 채널 마스킹**: `SoundscapeDataset` 은 `gt[:n_fg]` 만 채우고 나머지 (`num_output_channels - n_fg`) 채널은 0 으로 패딩합니다. 그대로 평균 내면 정의되지 않은 si_sdr / snr 값이 평균을 오염시키므로, `targets["num_fg_labels"]` 와 `arange(C)` 를 비교해 만든 `(B, C)` boolean mask 로 활성 채널만 골라 계산. 완전 벡터화 (배치 루프 없음).

### 2-4. ModelCheckpoint / EarlyStopping

```python
checkpoint_cb = ModelCheckpoint(
    dirpath=save_dir,                # configs/tse/orange_pi.yaml: "runs/tse"
    monitor=monitor,                 # "val/loss"
    mode=mode,                       # "min"
    save_top_k=ckpt_cfg.get("save_top_k", 1),
    filename="best-{epoch:02d}",
)

if es_cfg.get("patience", 0) > 0:
    callbacks.append(EarlyStopping(monitor=monitor, mode=mode,
                                   patience=patience, verbose=True))
```

---

## 3. Fabric 백엔드 — `FabricTrainerBackend`

`src/trainer/fabric.py:125` — Lightning Fabric 으로 **수동 학습 루프** 를 제공.

### 3-1. fit() 의 핵심

```
fabric = Fabric(accelerator="auto", devices="auto", precision="32-true")
fabric.launch()

model, optimizer = fabric.setup(model, optimizer)
train_loader, val_loader = fabric.setup_dataloaders(train_loader, val_loader)

for epoch in range(max_epochs):
    train_loss = _train_one_epoch(model, train_loader, loss_fn, optimizer, fabric, grad_clip)
    val_loss, val_metrics = _validate_one_epoch(model, val_loader, loss_fn, fabric, metrics_fn)

    # scheduler step
    if isinstance(scheduler, ReduceLROnPlateau):
        scheduler.step(current_metric)
    else:
        scheduler.step()

    # best tracking + checkpoint
    if is_better:
        save_checkpoint(...); patience_counter = 0
    else:
        patience_counter += 1
        if patience > 0 and patience_counter >= patience:
            break  # early stop
```

### 3-2. `_train_one_epoch`

`src/trainer/fabric.py:20`

```python
for batch in train_loader:
    optimizer.zero_grad()
    inputs, targets = batch
    outputs = model(inputs)
    loss = loss_fn(outputs, targets)
    fabric.backward(loss)
    if grad_clip:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()

# Aggregate across devices (DDP)
gathered_loss = fabric.all_gather(...)
```

### 3-3. 분산 학습 지원

`fabric.barrier()`, `fabric.all_gather()` 가 모든 device 의 loss/metric 을 합산. `model.module if hasattr(model, "module")` 로 DDP/FSDP wrapper unwrap (`fabric.py:233`).

---

## 4. Backend 선택 가이드

| 상황 | 권장 backend | 이유 |
|---|---|---|
| 일반 학습 (단일 GPU, 단일 노드) | **lightning** | 콜백/체크포인트/로깅 자동 |
| 멀티 GPU (DDP) | lightning 또는 fabric | 둘 다 OK; lightning 이 더 자동 |
| 커스텀 학습 루프 (예: GAN, RL) | fabric | 수동 루프 제어 |
| 디버깅 / 단일 step trace | fabric | print/breakpoint 삽입 쉬움 |

본 레포 default 는 lightning (`configs/tse/orange_pi.yaml:33`).

---

## 5. SED 도 동일 trainer 사용 (commit `7a3ad68` 이후)

이전 버전의 본 문서는 SED 가 inline loop 를 사용한다고 적혀 있었으나, **시그니처 통일 후 TSE 와 동일하게 `trainer.fit()` 으로 통합**되었습니다.

```
SED train flow (현재):
  cfg → SoundscapeDataset(task="sed")
       → ASTModel (HuggingFace AST, freeze_model() 호출)
       → _build_optimizer(): layer-wise AdamW (encoder_lr=1e-5, head_lr=1e-3)
                             ← 옵티마이저는 SED 전용 함수 유지 (param_groups 분리)
       → _build_scheduler(): scheduler_name dotted-path (예: CosineAnnealingWarmRestarts)
       → _build_loss_fn(): BCE(outputs["output"], inputs["labels"])
       → _build_metrics_fn(): ClassificationMetrics
       → trainer.fit(model, ..., loss_fn, optimizer, scheduler, cfg, metrics_fn)
            ↑ Lightning 또는 Fabric (yaml backend 키)
```

`_build_optimizer` 만 layer-wise 가 필요해 SED 별 함수로 유지되고, 나머지 (loss/metrics/scheduler/trainer) 는 모두 TSE 와 같은 패턴.

다음: [모델 구조 →](./model-structure.md)
