# 03-4. wandb 사용 예제

[← 학습 개요](./overview.md)

본 문서는 Lightning / Fabric 백엔드에서 **Weights & Biases (wandb)** 를 붙이는 코드 예제를 모아둡니다.

---

## 1. 설치 및 로그인

```bash
pip install wandb
wandb login   # https://wandb.ai/authorize 에서 토큰 받아 입력
```

또는 환경 변수로:

```bash
export WANDB_API_KEY=<your_token>
export WANDB_PROJECT=fine_grained_soundscape
export WANDB_ENTITY=<your_team_or_username>
```

---

## 2. Lightning 백엔드에 wandb 붙이기

### 2-1. 가장 간단한 방법 — config 의 `logger` 키만 채우기

`src/trainer/lightning.py:143` 에서 logger 가 결정됩니다:

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

학습 코드 진입점에서 `WandbLogger` 를 만들어 config 에 주입:

```python
# src/tse/train.py 의 main() 안에서, trainer.fit() 호출 직전
from pytorch_lightning.loggers import WandbLogger

wandb_logger = WandbLogger(
    project="fine_grained_soundscape",
    name=f"orange_pi_run01",
    config=cfg,                    # 전체 yaml config 자동 기록
    save_dir="runs/tse",
    log_model=True,                # checkpoint 도 wandb artifact 로 업로드
)
cfg["logger"] = wandb_logger       # ← LightningTrainerBackend 가 읽어 감

trainer.fit(model, train_loader, val_loader, loss_fn, optimizer, scheduler, cfg, metrics_fn)
```

### 2-2. 실행

```bash
python -m src.tse.train \
    --config configs/tse/orange_pi.yaml \
    --data_dir /path/to/data
```

자동으로 다음이 wandb 에 기록됩니다:

- `_LitWrapper.training_step` 의 `self.log("train/loss", loss, ...)`
- `_LitWrapper.validation_step` 의 `self.log("val/loss", ...)`, `self.log("val/si_sdri", ...)`, `self.log("val/snri", ...)`
- `LearningRateMonitor` 콜백의 lr-vs-step 곡선
- `ModelCheckpoint` 의 best ckpt (artifact 로 업로드, `log_model=True` 시)

---

## 3. Lightning + wandb 고급 사용

### 3-1. 오디오 샘플 로깅

매 N epoch 마다 validation 의 일부 mixture/target/estimate 를 wandb 에 audio 로 업로드:

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

        # 첫 batch 의 첫 sample 만 epoch 간격으로 audio 업로드
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

### 3-2. 스펙트로그램 이미지 로깅

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

### 3-3. gradient histogram 자동 로깅

```python
wandb_logger = WandbLogger(
    project="fine_grained_soundscape",
    log_model=True,
)
wandb_logger.watch(
    lit_model,
    log="gradients",          # "parameters", "gradients", "all"
    log_freq=100,             # step 단위
)
```

---

## 4. Fabric 백엔드에 wandb 붙이기

`src/trainer/fabric.py` 는 Lightning Trainer 가 아닌 수동 루프이므로 **wandb 를 직접 호출** 합니다.

```python
# src/trainer/fabric.py 의 fit() 안 (예시)
import wandb

class FabricTrainerBackend(TrainerBackend):
    def fit(self, model, train_loader, val_loader, loss_fn,
            optimizer, scheduler, config, metrics_fn=None):
        # ... 기존 코드 ...

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
        # ... 기존 코드 ...
```

config 에서 활성화:

```yaml
# configs/tse/orange_pi.yaml
training:
  backend: "fabric"
  use_wandb: true
  wandb_project: "fine_grained_soundscape"
  run_name: "orange_pi_fabric_run01"
```

---

## 5. CLI 에서 직접 실행 (스크립트 수정 없이)

`src/tse/train.py` 코드를 건드리지 않고 wandb 를 켜고 싶다면, **환경 변수 + Lightning 의 자동 감지** 를 활용:

```bash
# 1) wandb 환경 변수 셋업
export WANDB_API_KEY=<token>
export WANDB_PROJECT=fine_grained_soundscape
export WANDB_RUN_NAME=orange_pi_run01

# 2) PyTorch Lightning 의 wandb 통합 켜기 (logger=True 일 때 자동으로 picked up 되지 않음)
#    → 코드 수정이 최소한이라도 필요. src/tse/train.py 에 다음 한 줄 추가:
#       cfg["logger"] = WandbLogger()  # main() 안에서
```

> **참고**: Lightning 의 `pl.Trainer(logger=True)` 만 으로는 wandb 가 자동 활성화되지 않습니다. 명시적으로 `WandbLogger` 인스턴스를 넘겨야 합니다.

---

## 6. 추적 권장 metric

| 카테고리 | metric key | 출처 |
|---|---|---|
| **학습 손실** | `train/loss` | `_LitWrapper.training_step` |
| **검증 손실** | `val/loss` | `_LitWrapper.validation_step` |
| **TSE quality** | `val/si_sdri` | `_build_metrics_fn` (`src/tse/train.py:127`) |
| | `val/snri` | 동일 |
| **학습률** | `lr-AdamW` | `LearningRateMonitor` |
| **gradient norm** | `grad_2norm` | `wandb.watch(log="gradients")` |
| **체크포인트** | `epoch`, `best_metric` | `ModelCheckpoint` |

---

## 7. 가속기 (accelerator) 옵션

Lightning / Fabric 모두 `accelerator="auto"`, `devices="auto"` 가 기본값 (`src/trainer/lightning.py:148-149`, `src/trainer/fabric.py:166-167`). 명시적으로 지정하려면:

### 7-1. Lightning

```python
# src/trainer/lightning.py 의 fit() 의 _extra_kwargs 로 전달
trainer = create_trainer(
    backend="lightning",
    accelerator="gpu",          # "cpu", "gpu", "tpu", "auto"
    devices=2,                  # 또는 [0, 1] 또는 "auto"
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
    precision="16-mixed",       # config: precision="32-true" 기본
)
```

> 단일 Colab GPU 환경에서는 `accelerator="auto", devices="auto"` 그대로 두면 자동으로 `gpu`, `1` 로 잡힙니다.

