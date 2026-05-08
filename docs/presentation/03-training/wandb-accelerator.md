# 03-4. wandb Usage Examples

[← Training Overview](./overview.md)

This document collects code examples for hooking **Weights & Biases (wandb)** into the Lightning / Fabric backends.

---

## 1. Install and Log In

```bash
pip install wandb
wandb login   # paste a token obtained at https://wandb.ai/authorize
```

Or via environment variables:

```bash
export WANDB_API_KEY=<your_token>
export WANDB_PROJECT=fine_grained_soundscape
export WANDB_ENTITY=<your_team_or_username>
```

---

## 2. Attaching wandb to the Lightning Backend

### 2-1. Simplest Approach — Set the `logger` Key in Config

The logger is selected at `src/trainer/lightning.py:143`:

```python
pl_logger = config.get("logger", True)
trainer = pl.Trainer(
    max_epochs=max_epochs,
    accelerator="auto",
    devices="auto",
    callbacks=callbacks,
    logger=pl_logger,
    **self._extra_kwargs,
)
```

Build a `WandbLogger` at the training entry point and inject it into the config:

```python
# Inside main() of src/tse/train.py, just before calling trainer.fit()
from pytorch_lightning.loggers import WandbLogger

wandb_logger = WandbLogger(
    project="fine_grained_soundscape",
    name=f"orange_pi_run01",
    config=cfg,                    # full yaml config is logged automatically
    save_dir="runs/tse",
    log_model=True,                # also upload checkpoints as wandb artifacts
)
cfg["logger"] = wandb_logger       # ← LightningTrainerBackend will pick this up

trainer.fit(model, train_loader, val_loader, loss_fn, optimizer, scheduler, cfg, metrics_fn)
```

### 2-2. Run It

```bash
python -m src.tse.train \
    --config configs/tse/orange_pi.yaml \
    --data_dir /path/to/data
```

The following are logged to wandb automatically:

- `self.log("train/loss", loss, ...)` from `_LitWrapper.training_step`
- `self.log("val/loss", ...)`, `self.log("val/si_sdri", ...)`, `self.log("val/snri", ...)` from `_LitWrapper.validation_step`
- The lr-vs-step curve from the `LearningRateMonitor` callback
- The best checkpoint from `ModelCheckpoint` (uploaded as an artifact when `log_model=True`)

---

## 3. Advanced Lightning + wandb Usage

### 3-1. Logging Audio Samples

Upload a subset of validation mixture/target/estimate audio to wandb every N epochs:

```python
import wandb
import torch

class _LitWrapper(pl.LightningModule):
    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.model(inputs)
        est = outputs["output"]; gt = targets["target"]
        loss = self.loss_fn(est=est, gt=gt).mean()
        self.log("val/loss", loss, prog_bar=True, sync_dist=True)

        # Upload audio for the first sample of the first batch at fixed epoch intervals
        if batch_idx == 0 and self.current_epoch % 10 == 0:
            sr = 16000
            mix_np = inputs["mixture"][0].mean(0).cpu().numpy()       # mono
            est_np = est[0, 0].cpu().numpy()
            gt_np  = gt[0, 0].cpu().numpy()

            self.logger.experiment.log({
                "val/audio/mixture":  wandb.Audio(mix_np, sample_rate=sr, caption=f"epoch={self.current_epoch}"),
                "val/audio/estimate": wandb.Audio(est_np, sample_rate=sr, caption=f"epoch={self.current_epoch}"),
                "val/audio/target":   wandb.Audio(gt_np,  sample_rate=sr, caption=f"epoch={self.current_epoch}"),
            })
```

### 3-2. Logging Spectrogram Images

```python
import matplotlib.pyplot as plt
import librosa

if batch_idx == 0 and self.current_epoch % 20 == 0:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, sig, title in zip(axes, [mix_np, est_np, gt_np], ["mix", "est", "gt"]):
        S = librosa.amplitude_to_db(np.abs(librosa.stft(sig, n_fft=1024, hop_length=256)))
        librosa.display.specshow(S, sr=sr, ax=ax, x_axis="time", y_axis="hz")
        ax.set_title(title)
    plt.tight_layout()
    self.logger.experiment.log({"val/spectrograms": wandb.Image(fig)})
    plt.close(fig)
```

### 3-3. Automatic Gradient Histogram Logging

```python
wandb_logger = WandbLogger(
    project="fine_grained_soundscape",
    log_model=True,
)
wandb_logger.watch(
    lit_model,
    log="gradients",          # "parameters", "gradients", "all"
    log_freq=100,             # in steps
)
```

---

## 4. Attaching wandb to the Fabric Backend

`src/trainer/fabric.py` is a manual loop rather than a Lightning Trainer, so **wandb must be called directly**.

```python
# Inside fit() of src/trainer/fabric.py (example)
import wandb

class FabricTrainerBackend(TrainerBackend):
    def fit(self, model, train_loader, val_loader, loss_fn,
            optimizer, scheduler, config, metrics_fn=None):
        # ... existing code ...

        if config.get("use_wandb", False) and fabric.global_rank == 0:
            wandb.init(
                project=config.get("wandb_project", "fine_grained_soundscape"),
                name=config.get("run_name", None),
                config=config,
            )
            wandb.watch(model, log="gradients", log_freq=100)

        for epoch in range(max_epochs):
            train_loss = _train_one_epoch(model, train_loader, loss_fn, optimizer, fabric, grad_clip)
            val_loss, val_metrics = _validate_one_epoch(model, val_loader, loss_fn, fabric, metrics_fn)

            if fabric.global_rank == 0 and config.get("use_wandb", False):
                wandb.log({
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "val/loss": val_loss,
                    "lr": optimizer.param_groups[0]["lr"],
                    **{f"val/{k}": v for k, v in val_metrics.items()},
                })
        # ... existing code ...
```

Enable in config:

```yaml
# configs/tse/orange_pi.yaml
training:
  backend: "fabric"
  use_wandb: true
  wandb_project: "fine_grained_soundscape"
  run_name: "orange_pi_fabric_run01"
```

---

## 5. Running Directly from the CLI (Without Editing Scripts)

If you want to enable wandb without touching `src/tse/train.py`, leverage **environment variables + Lightning's auto-detection**:

```bash
# 1) Set up wandb environment variables
export WANDB_API_KEY=<token>
export WANDB_PROJECT=fine_grained_soundscape
export WANDB_RUN_NAME=orange_pi_run01

# 2) Enable PyTorch Lightning's wandb integration (it is NOT auto picked up when logger=True)
#    → A minimal code change is still required. Add this single line in src/tse/train.py:
#       cfg["logger"] = WandbLogger()  # inside main()
```

> **Note**: `pl.Trainer(logger=True)` alone does not auto-enable wandb in Lightning. You must explicitly pass a `WandbLogger` instance.

---

## 6. Recommended Metrics to Track

| Category | metric key | Source |
|---|---|---|
| **Training loss** | `train/loss` | `_LitWrapper.training_step` |
| **Validation loss** | `val/loss` | `_LitWrapper.validation_step` |
| **TSE quality** | `val/si_sdri` | `_build_metrics_fn` (`src/tse/train.py:127`) |
| | `val/snri` | same |
| **Learning rate** | `lr-AdamW` | `LearningRateMonitor` |
| **Gradient norm** | `grad_2norm` | `wandb.watch(log="gradients")` |
| **Checkpoints** | `epoch`, `best_metric` | `ModelCheckpoint` |

---

## 7. Accelerator Options

Both Lightning and Fabric default to `accelerator="auto"`, `devices="auto"` (`src/trainer/lightning.py:148-149`, `src/trainer/fabric.py:166-167`). To set them explicitly:

### 7-1. Lightning

```python
# Pass through _extra_kwargs of fit() in src/trainer/lightning.py
trainer = create_trainer(
    backend="lightning",
    accelerator="gpu",          # "cpu", "gpu", "tpu", "auto"
    devices=2,                  # or [0, 1] or "auto"
    strategy="ddp",             # "ddp", "ddp_spawn", "fsdp", "auto"
    precision="16-mixed",       # "32-true", "16-mixed", "bf16-mixed"
    gradient_clip_val=1.0,
)
```

### 7-2. Fabric

```python
trainer = create_trainer(
    backend="fabric",
    accelerator="gpu",
    devices=2,
    strategy="ddp",
    precision="16-mixed",       # config: precision="32-true" by default
)
```

> In a single Colab GPU environment, leaving `accelerator="auto", devices="auto"` resolves automatically to `gpu` and `1`.
