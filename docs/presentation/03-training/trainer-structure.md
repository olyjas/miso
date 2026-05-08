# 03-2. Trainer Structure

[← Training Overview](./overview.md)

---

## 1. Interface — `TrainerBackend` ABC

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

Thanks to this interface, the model and dataset code can be written independently of the backend.

### Factory Function

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

## 2. Lightning Backend — `LightningTrainerBackend`

`src/trainer/lightning.py:89`

### 2-1. Structure

```
LightningTrainerBackend
  │
  ├─ _LitWrapper (pl.LightningModule)             # src/trainer/lightning.py:23
  │     ├ training_step:   loss = loss_fn(est, gt).mean()
  │     │                  self.log("train/loss", ...)
  │     ├ validation_step: same + self.log("val/loss", ...)
  │     │                  + self.log("val/si_sdri"), ("val/snri")
  │     └ configure_optimizers: {"optimizer": ..., "lr_scheduler": {...}}
  │
  └─ fit():
        callbacks = [ModelCheckpoint(monitor="val/loss", mode="min", ...),
                     LearningRateMonitor(),
                     # EarlyStopping(patience=es_cfg.patience) if patience>0 ]
        trainer = pl.Trainer(
            max_epochs=200,
            accelerator="auto",          # auto-select GPU/CPU
            devices="auto",              # all available devices
            callbacks=callbacks,
            logger=pl_logger,            # config["logger"] or True
            **self._extra_kwargs,        # user-defined options
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

> **Unified signature (commit `7a3ad68`)**: the trainer no longer extracts dict entries — it calls the loss with the 3-arg form `(outputs, targets, inputs)`. Pulling tensors out of these dicts is the responsibility of task-specific adapters (`src/tse/train.py:_build_loss_fn` / `src/sed/train.py:_build_loss_fn`).

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

`metrics_fn` is the adapter built in `src/tse/train.py:_build_metrics_fn` (or `src/sed/train.py:_build_metrics_fn` for SED).

> **TSE metric — padded channel masking**: `SoundscapeDataset` only fills `gt[:n_fg]` and pads the remaining (`num_output_channels - n_fg`) channels with zeros. Averaging directly would let undefined si_sdr / snr values from the padded slots contaminate the mean, so a `(B, C)` boolean mask is built by comparing `targets["num_fg_labels"]` with `arange(C)` and used to select active channels only. Fully vectorized — no per-sample loop.

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

## 3. Fabric Backend — `FabricTrainerBackend`

`src/trainer/fabric.py:125` — provides a **manual training loop** built on Lightning Fabric.

### 3-1. Core of fit()

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

### 3-3. Distributed Training Support

`fabric.barrier()` and `fabric.all_gather()` aggregate loss/metrics across all devices. The DDP/FSDP wrapper is unwrapped via `model.module if hasattr(model, "module")` (`fabric.py:233`).

---

## 4. Backend Selection Guide

| Scenario | Recommended backend | Reason |
|---|---|---|
| General training (single GPU, single node) | **lightning** | Automatic callbacks, checkpointing, logging |
| Multi-GPU (DDP) | lightning or fabric | Both work; lightning is more automatic |
| Custom training loops (e.g., GAN, RL) | fabric | Manual loop control |
| Debugging / single-step tracing | fabric | Easier to insert print/breakpoint |

The default backend in this repo is lightning (`configs/tse/orange_pi.yaml:33`).

---

## 5. SED Uses the Same Trainer (since commit `7a3ad68`)

A previous version of this document stated that SED used an inline loop, but **after the signature was unified, SED converged with TSE on the same `trainer.fit()` path**.

```
SED train flow (current):
  cfg → SoundscapeDataset(task="sed")
       → ASTModel (HuggingFace AST, freeze_model() called)
       → _build_optimizer(): layer-wise AdamW (encoder_lr=1e-5, head_lr=1e-3)
                             ← optimizer remains an SED-specific function (separate param_groups)
       → _build_scheduler(): scheduler_name dotted-path (e.g., CosineAnnealingWarmRestarts)
       → _build_loss_fn(): BCE(outputs["output"], inputs["labels"])
       → _build_metrics_fn(): ClassificationMetrics
       → trainer.fit(model, ..., loss_fn, optimizer, scheduler, cfg, metrics_fn)
            ↑ Lightning or Fabric (selected via the yaml backend key)
```

Only `_build_optimizer` is kept SED-specific because layer-wise grouping is required; everything else (loss / metrics / scheduler / trainer) follows the same pattern as TSE.

Next: [Model Structure →](./model-structure.md)
