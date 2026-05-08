# 03-3. Model Structure (TFGridNet Large)

[← Training Overview](./overview.md) · [← Trainer Structure](./trainer-structure.md)

This document summarizes the model structure from a training perspective. For inference branching and STFT details, see [01-2. TSE Pipeline](../01-architecture/tse-pipeline.md).

---

## 1. Model Call Sequence During Training

```
batch (DataLoader)                Lightning training_step
  │                                 │
  │ inputs = {                       │  inputs, targets = batch
  │   "mixture":     (B, 2, 80000),  │  outputs = self.model(inputs)
  │   "label_vector":(B, 20),        │  est = outputs["output"]
  │ }                                │  gt  = targets["target"]
  │ targets = {                      │  loss = loss_fn(est, gt)
  │   "target": (B, 5, 80000), ...   │
  │ }                                │
  ▼                                 ▼
                          Net.forward(inputs)
                          src/tse/net.py:216
                            │
                            │ (Note: forward reads "mixture" and
                            │  "embedding" from inputs. The training
                            │  code maps label_vector → embedding,
                            │  either at the call site or inside the dataset.)
                            ▼
                          predict(x, embedding, state, pad)
                            ├ mod_pad
                            ├ extract_features (STFT)
                            ├ tfgridnet(x, emb, state)
                            └ synthesis (iSTFT + overlap-add)
                            ▼
                          {"output": (B, 5, 80000),
                           "next_state": dict}
```

---

## 2. Class Hierarchy

```
src.tse.net.Net                              ← STFT + separator + iSTFT wrapper
├ self.tfgridnet = MultiFiLMGuidedTFNet      ← separator core
│   ├ self.conv = nn.Conv2d(num_inputs=4, latent_dim=32, k=3x3)
│   ├ self.embedding = None                  ← None because embedding_dim=0
│   ├ self.film_layers = nn.ModuleDict({
│   │     "film_layer_0": FiLM(D=32, emb=20),
│   │     "film_layer_1": FiLM(D=32, emb=20),
│   │     ...
│   │     "film_layer_5": FiLM(D=32, emb=20),
│   │   })   ← film_preset="all" → 6 FiLM layers
│   ├ self.blocks = nn.ModuleList([
│   │     GridNetBlock(D=32, n_freqs=129, hidden=64),  × 6
│   │   ])
│   └ self.deconv = nn.ConvTranspose2d(D=32, n_srcs*2=10, k=3x3)
│
├ self.analysis_window  (rect, length nfft=256)
└ self.synthesis_window (perfect reconstruction window)
```

### 2-1. Parameter Count (Measured)

| Module | params | Ratio |
|---|---|---|
| `tfgridnet.conv` | 1,184 | 0.2% |
| `tfgridnet.film_layers` (6 × FiLM) | 8,064 | 1.6% |
| `tfgridnet.blocks` (6 × GridNetBlock) | **489,600** | **97.6%** |
| `tfgridnet.deconv` | 2,890 | 0.6% |
| **Total** | **501,738 (0.502 M)** | 100% |

> **Practical footprint**: 0.5M params × 4 bytes (fp32) ≈ **2 MB** model file — very lightweight.

### 2-2. Inside a GridNetBlock (single block)

```
GridNetBlock(latent_dim=32, n_freqs=129, hidden_channels=64)
  │
  ├ intra-frame branch (processes the frequency axis)
  │   ├ LayerNorm(32)                                    →   64 params
  │   ├ LSTM(in=32, hid=64, layers=1, bidir=True)        ← 49,920 params
  │   └ Linear(128 → 32)                                 →  4,128 params
  │   subtotal: ~54,000
  │
  └ inter-frame branch (processes the time axis, causal)
      ├ LayerNorm(32)                                    →     64 params
      ├ LSTM(in=32, hid=64, layers=1, bidir=False)       ← 24,960 params
      └ Linear(64 → 32)                                  →  2,080 params
      subtotal: ~27,000

Total per block ≈ 81,600 params. × 6 blocks = 489,600 params.
```

### 2-3. A Single FiLM

```
FiLM(input_channels=32, embedding_channels=20)
  ├ a: nn.Linear(20 → 32)   →  672 params
  └ b: nn.Linear(20 → 32)   →  672 params
  total: 1,344 params

× 6 layers = 8,064 params.

forward(x, emb):
  a = self.a(emb)  # (B, 32)
  b = self.b(emb)  # (B, 32)
  unsqueeze to (B, 32, 1, 1)
  return x * a + b   # (B, 32, T, F)
```

`src/tse/film.py:11`

---

## 3. Loss — `MultiResoFuseLoss`

`src/tse/loss.py:MultiResoFuseLoss`

```
loss = MultiResolutionSTFTLoss(auraloss) + l1_ratio * L1Loss
       │                                    │
       │ Multi-scale STFT loss              │ Time-domain L1 loss
       │ (perceptually meaningful           │
       │  spectral distance)                │
       │                                    │
       └─ averaged across multiple          └─ l1_ratio=10 (config)
          resolutions (various nfft, hop)
```

The default settings of `MultiResolutionSTFTLoss` follow the [auraloss defaults](https://github.com/csteinmetz1/auraloss).

---

## 4. Optimizer / Scheduler

`src/tse/train.py:271-277`:

```python
# dotted-path build — class and hyperparameters are all driven by yaml
optimizer = _import_attr(tc["optimizer_name"])(
    model.parameters(),
    **tc.get("optimizer_params", {}),
)
scheduler = _import_attr(tc["scheduler_name"])(
    optimizer,
    **tc.get("scheduler_params", {}),
)
```

Key matching in `configs/tse/orange_pi.yaml:43-50`:

```yaml
optimizer_name: "torch.optim.AdamW"
optimizer_params: { lr: 0.001, weight_decay: 0.0 }
scheduler_name: "torch.optim.lr_scheduler.ReduceLROnPlateau"
scheduler_params: { factor: 0.5, patience: 5 }
```

- `monitor` is `val/loss` (config `checkpointing.monitor`).
- ReduceLROnPlateau multiplies lr by 0.5 when `val/loss` does not improve for 5 epochs.
- Gradient clipping: `grad_clip=1.0` (set via Lightning's `gradient_clip_val`; in this repo it becomes active when passed through additional kwargs to `pl.Trainer(...)` or via `_extra_kwargs`).

---

## 5. One Step in the Training Loop

```
[ DataLoader prefetch (worker × 8) ]
        │
        ▼
batch = (inputs, targets)        ← (8, 2, 80000), (8, 5, 80000)
        │
        ▼
optimizer.zero_grad()
        │
        ▼
outputs = model(inputs)          ← Forward MACs 4.61G/sample × 8 ≈ 37G MACs
        │
        ▼
loss = MR-STFT(est, gt) + 10 * L1(est, gt)
        │
        ▼
loss.backward()                  ← Backward ≈ 2 × forward = ~74G MACs
        │
        ▼
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        │
        ▼
optimizer.step()                 ← AdamW: w -= lr * m_hat / (sqrt(v_hat) + eps)
        │
        ▼
self.log("train/loss", loss)     ← sent to Lightning's default logger
```

**Theoretical compute per step**: ~111G MACs (forward 37G + backward ~74G), ≈ 0.22 TFLOP.

> ⚠️ **Memory is not light.** Although the parameter count is small (0.5 M), LSTM backward activations push FP32 training at batch=8 to **~34 GB VRAM**.

Next: [wandb Usage Examples →](./wandb-accelerator.md)
