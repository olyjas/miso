# 04. Training GPU Resource Estimate — TFGridNet Large

[← Back to README](../README.md)

This document estimates the GPU VRAM and wall-clock time required to train **TFGridNet Large** (`configs/tse/orange_pi.yaml`), based on **measurements plus extrapolation**.

---

## 🚨 Key Takeaways (TL;DR)

| Item | Value |
|---|---|
| Model parameters | 0.502 M (very small) |
| **B=8 FP32 training VRAM** | **~34 GB** (estimated) |
| **B=8 AMP FP16 training VRAM** | **~18 GB** (estimated) |
| Recommended GPU (B=8 FP32) | **L40S 48GB × 2 (DDP)**, or a single A100 40GB (tight) |
| Colab recommendation | Pro+ A100 40GB (B=8 feasible), L4 24GB (requires AMP) |
| Colab Free (T4 16GB) | **Not feasible** (only B=2 fits, training is impractical) |

> 🔑 **Why does a 0.5M-parameter model need over 30 GB?**
> TFGridNet's LSTMs must **store hidden states for every time step** to compute the backward pass. The intra-frame LSTM operates on an effective batch of `B×T` (= 8×836 = 6,688), so activation memory ends up **hundreds of times larger than the model itself**.

---

## 1. Measurements (RTX 4070 Ti SUPER, 16 GB, FP32)

These numbers are peak memory observed across one full forward + backward + `optimizer.step` cycle while training with `MultiResoFuseLoss`.

### 1-1. VRAM Usage (Measured)

| Batch | FP32 + MR-STFT | AMP FP16 + MR-STFT | Notes |
|---|---|---|---|
| 1 | **4.23 GB** ✅ measured | **2.22 GB** ✅ measured | |
| 2 | **8.69 GB** ✅ measured | **4.54 GB** ✅ measured | |
| 3 | OOM (16GB) | — | |
| 4 | (~17.4 GB est.) | **9.04 GB** ✅ measured | |
| 8 | **~34.8 GB est.** | ~18.1 GB est. | **User report: a single L40S (48GB) is insufficient** |
| 16 | ~69.6 GB est. | ~36.2 GB est. | |

> The extrapolation is reliable because per-sample memory is nearly linear (~4.3 GB/sample FP32, ~2.3 GB/sample AMP). However, framework overhead, NCCL/DDP buffers, and the multi-resolution intermediate tensors of MR-STFT add about **+1-2 GB margin**.

### 1-2. Step Time (Measured)

| Setting | Step time | Throughput |
|---|---|---|
| FP32 B=2 | **139.7 ms** | 14.3 samples/s |
| AMP B=4 | **203.9 ms** | 19.6 samples/s |

- GPU: RTX 4070 Ti SUPER (FP32 peak 44.1 TFLOPS, FP16 peak 88.2 TFLOPS)
- Because the model is LSTM-heavy, SM utilization is low (~3% of peak — a known limitation of cuDNN LSTM)
- AMP is slightly slower per step in this setting, but the **VRAM savings allow a larger batch, so samples/s is higher**

---

## 2. Why Memory Is So Large — TFGridNet's LSTM Activations

```
Input (B, 2, 80000) → STFT → (B, 4, T=836, F=129) → reshape → (B, T, F, D=32)
                                                                       │
6 × GridNetBlock, each:                                                 │
  ├ intra-frame LSTM:                                                  │
  │   reshape → (B*T, F, D) = (8*836, 129, 32)                         │
  │   bidir LSTM(C=32 → H=64) → store every step's hidden state for backward │
  │   activation: ~1.0 GB per block per batch element                  │
  │                                                                    │
  ├ inter-frame LSTM (causal):                                         │
  │   transpose → (B*F, T, D) = (8*129, 836, 32)                       │
  │   LSTM(C=32 → H=64) → store every step's hidden state              │
  │   activation: ~0.8 GB per block per batch element                  │
  │                                                                    │
  └ ───────────                                                        │
       per-block activation total: ~1.8 GB × 6 blocks = ~10.8 GB       │
       + STFT + Conv2d + FiLM + autograd graph overhead                │
                                                                       │
                                       Total: ~25 GB (FP32, B=8) est.
```

**Key point**: the intra-frame LSTM has an effective batch of `B*T` = `8*836` = 6,688. Because RNNs must keep **every time step's hidden + cell state** for backprop, activations explode.

---

## 3. Colab GPU Compatibility Matrix

| GPU | VRAM | FP32 TFLOPS | TF32 | FP16 | B=8 FP32 | B=8 AMP | B=4 AMP | B=2 FP32 |
|---|---|---|---|---|---|---|---|---|
| **T4** (Free) | 16 GB | 8.1 | — | 65.1 | ❌ | ❌ | ✅ (~9 GB) | ✅ (8.7 GB) |
| **V100** (Pro) | 16 GB | 14.0 | — | 112 | ❌ | ❌ | ✅ | ✅ |
| **L4** (Pro+) | 24 GB | 30.3 | 121 | 242 | ❌ | ✅ (~18 GB) | ✅ | ✅ |
| **A100 40 GB** (Pro+) | 40 GB | 19.5 | 156 | 312 | ⚠️ tight (~35 GB) | ✅ | ✅ | ✅ |
| **L40S 48 GB** (reference) | 48 GB | 91.6 | 183 | 362 | ⚠️ insufficient margin | ✅ | ✅ | ✅ |
| **L40S × 2 DDP** (user environment) | 96 GB | × 2 | × 2 | × 2 | ✅ B=8 distributed | ✅ B=16 distributed | ✅ | ✅ |
| **A100 80 GB** | 80 GB | 19.5 | 156 | 312 | ✅ (~35 GB) | ✅ B=16 | ✅ | ✅ |

> ⚠️ "tight" / "insufficient margin" means OOM is plausible once framework overhead, DataLoader pinned memory, NCCL, etc. are added. **As reported by users, a single L40S is not enough; 2 × L40S DDP is the safe choice.**

### 3-1. Colab Free (T4 16 GB)

```
❌ Training is not recommended.
   - B=2 is the maximum (8.7 GB) — only 1/4 of the paper's batch=8
   - Even with AMP, B=4 is the ceiling
   - 12 h session timeout → 200 epochs is impossible
   - Disk limit (~100 GB) → cannot hold the full dataset (110 GB)
Recommended use: inference verification on the mini dataset (~250 MB) only
```

### 3-2. Colab Pro (V100 / T4 / L4 mixed)

```
🟡 Training is feasible but requires a smaller batch size.
   - V100 16 GB: same limits as T4 (B=2 FP32, B=4 AMP)
   - L4 24 GB:    B=4 FP32, B=8 AMP fits
   - 24 h sessions → 200 epochs require split runs across multiple days
Recommended batch_size: AMP B=4 (or B=8 on L4)
```

### 3-3. Colab Pro+ (A100 40 GB)

```
✅ Recommended environment.
   - B=8 FP32 fits (~35 GB, 5 GB margin)
   - B=16 AMP fits (~36 GB)
   - TF32 auto-enabled for matmul speedup (modest)
   - 24 h session limit still applies → 200 epochs require split runs
   - torch.compile() can offer additional speedup
```

---

## 4. 200-Epoch Wall-Clock Estimate

samples_per_epoch = 20,000 with batch_size = 8 → 2,500 steps/epoch, 500,000 steps total.

### 4-1. Step-Time Estimate Scaled From Measurements

Baseline: RTX 4070 Ti Super measurements (FP32 B=2 = 140 ms, AMP B=4 = 204 ms), scaled by FP32 TFLOPS.

| GPU | FP32 TFLOPS | step time (FP32 B=8 est.) | throughput | 1 epoch | 200 epochs |
|---|---|---|---|---|---|
| 4070 Ti Super (measured) | 44.1 | 560 ms (B=8 extrapolated) | 14.3 s/s | 23 min | **77 hours (3.2 d)** |
| T4 (B=8 OOM) | 8.1 | — (n/a) | — | — | — |
| V100 (B=8 OOM) | 14.0 | — | — | — | — |
| L4 24 GB (B=8 OOM) | 30.3 | — | — | — | — |
| A100 40 GB | 19.5 (FP32) | ~1,200 ms | 6.7 s/s | ~50 min | **~165 hours (7 d)** |
| A100 + TF32 | 156 (matmul) | ~400 ms est. | ~20 s/s | ~17 min | **~55 hours (2.3 d)** |
| L40S × 2 DDP | 91.6 × 2 | ~270 ms (effective) | ~30 s/s | ~11 min | **~37 hours (1.5 d)** |

> ⚠️ The estimates above only account for **GPU compute**. In practice, **data-loading bottlenecks (HRTF + LUFS)** add significant overhead.

### 4-2. Impact of the Data-Loading Bottleneck

CPU time spent processing one sample inside `SoundscapeDataset.__getitem__`:

- 3-9 audio file reads + resample → 50-200 ms
- LUFS normalize × 3-9 (each up to 100 iterations) → 200-500 ms
- HRTF convolution × 3-9 → 100-300 ms
- Total: **~500-1,000 ms per sample**

With `num_workers=8`, filling a batch of 8 takes roughly 0.5-1.0 s. **Slower than the GPU step** → data loading becomes the bottleneck.

| Environment | Step time (compute) | Step time (with data loading, est.) |
|---|---|---|
| Colab vCPU 2 (limited num_workers benefit) | — | **1.5-3.0 s/step** |
| Klone workstation (CPU 16+, fast SSD) | — | 0.7-1.2 s/step |
| L40S × 2 DDP, fast NVMe | 270 ms | 0.5-0.7 s/step |

→ Even on Colab Pro+ A100 40GB, **realistic 200-epoch runs take 5-10 days**.

### 4-3. With Acceleration Options (Reference)

| Technique | Memory savings | Time savings | Where to enable |
|---|---|---|---|
| AMP (FP16) | × 0.5 | × 1.0-1.2 | `pl.Trainer(precision="16-mixed")` |
| BF16 | × 0.5 | × 1.0-1.2 | `precision="bf16-mixed"` (A100/L40S) |
| `torch.compile()` | — | × 1.1-1.3 | `model = torch.compile(model)` |
| **Mixture pre-synthesis** | — | **× 5-10** | Pre-render mixtures to disk (not yet wired into `SoundscapeDataset`; the legacy `MisophoniaDataset.save_dataset()` is the only built-in option today) |
| DDP (multi-GPU) | × 1/N | × ~0.6/N | `devices=N, strategy="ddp"` |

> **The data-loading bottleneck is the biggest enemy**. Pre-synthesizing mixtures to disk shortens training time by 5-10× (at the cost of losing on-the-fly augmentation).

---

## 5. Recommended batch_size by Environment

| Environment | precision | batch_size | Notes |
|---|---|---|---|
| Colab Free (T4 16GB) | AMP | 4 | 200 epochs unrealistic; verification only |
| Colab Pro (V100 16GB) | AMP | 4 | 200 epochs feasible across split sessions |
| Colab Pro (L4 24GB) | AMP | 8 | Recommended |
| Colab Pro+ (A100 40GB) | FP32 | 8 | Paper default |
| Colab Pro+ (A100 40GB) | AMP | 16 | Higher throughput |
| L40S × 1 (48GB) | AMP | 8 | FP32 lacks margin |
| L40S × 2 DDP (96GB) | FP32 | 8 (effective 16) | User-validated environment |
| A100 80GB × 1 | FP32 | 16 | Recommended |

---

## 6. Snippet for Direct Measurement

We recommend running this snippet once in your own environment before launching a long training run:

```python
import torch, time, yaml, sys
sys.path.insert(0, '.')
from src.tse.net import Net
from src.tse.loss import MultiResoFuseLoss

device = torch.device('cuda')
cfg = yaml.safe_load(open('configs/tse/orange_pi.yaml'))
m = cfg['model']
model = Net(
    model_name=m['model_name'], block_model_name=m['block_model_name'],
    block_model_params=m['block_model_params'], speaker_dim=m['speaker_dim'],
    stft_chunk_size=m['stft_chunk_size'], stft_pad_size=m['stft_pad_size'],
    stft_back_pad=m['stft_back_pad'], num_input_channels=m['num_input_channels'],
    num_output_channels=m['num_output_channels'], num_layers=m['num_layers'],
    latent_dim=m['latent_dim'], embedding_params=m['embedding_params'],
    film_params=m['film_params'], use_first_ln=m['use_first_ln'],
).to(device)
model.train()
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
loss_fn = MultiResoFuseLoss(l1_ratio=10).to(device)

for B in [1, 2, 4, 8, 16]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        T = 16000 * 5
        x = torch.randn(B, 2, T, device=device)
        emb = torch.zeros(B, 20, device=device); emb[:, 0] = 1.0
        gt = torch.randn(B, 5, T, device=device)
        # warmup
        for _ in range(2):
            out = model({'mixture': x, 'embedding': emb})
            loss = loss_fn(est=out['output'], gt=gt).mean()
            loss.backward(); opt.step(); opt.zero_grad()
        torch.cuda.synchronize()
        # measure
        t0 = time.time()
        for _ in range(10):
            out = model({'mixture': x, 'embedding': emb})
            loss = loss_fn(est=out['output'], gt=gt).mean()
            loss.backward(); opt.step(); opt.zero_grad()
        torch.cuda.synchronize()
        dt = (time.time()-t0)/10
        peak = torch.cuda.max_memory_allocated()/1024**3
        print(f'B={B:2d}: {dt*1000:.0f} ms/step, peak={peak:.1f} GB, {B/dt:.1f} samples/s')
    except torch.cuda.OutOfMemoryError:
        print(f'B={B:2d}: OOM')
        torch.cuda.empty_cache()
        break
```

---

## 7. Measurement Environment (for Reproducibility)

| Item | Value |
|---|---|
| GPU | NVIDIA RTX 4070 Ti SUPER 16 GB |
| Driver | 550.54.14 (CUDA 12.4) |
| PyTorch | 2.6.0+cu124 |
| OS | Linux 5.15 |
| Python | 3.11 |
| Virtualenv | `uv venv .venv-mem-test` (uv 0.9.3) |
| Measurement command | the snippet above |
| Model config | `configs/tse/orange_pi.yaml` (D=32, H=64, B=6, 5-out) |
| Input length | 5 s @ 16 kHz (80,000 samples) |

Next: [20-class + noise notebook →](../04-samples/sound-samples.ipynb)
