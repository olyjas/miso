# 01-2. TSE Pipeline Detail (TFGridNet Large)

Reference model: `orange_pi` config (`configs/tse/orange_pi.yaml`)

[← Back to overview](./overview.md)

---

## 1. Top-Level Flow — `Net.forward`

`Net` (`src/tse/net.py:31`) is a wrapper that ties together the STFT front-end, the separator (MultiFiLMGuidedTFNet), and the iSTFT back-end.

```
inputs = {"mixture": (B, 2, 80000), "embedding": (B, 20)}
            │
            ▼
    ┌───────────────────────────────────────────────┐
    │ Net.forward(inputs, input_state, pad)         │
    │   src/tse/net.py:216                          │
    │                                               │
    │   1. init_buffers(B, device) (if no state)    │
    │   2. predict(x, embedding, state, pad)        │
    │      ├─ mod_pad: align to chunk_size + pad    │
    │      ├─ extract_features: STFT               │
    │      ├─ tfgridnet(x, emb, state)             │
    │      └─ synthesis: iFFT + overlap-add        │
    │                                               │
    │   returns:                                    │
    │     {"output": (B, 5, 80000),                 │
    │      "next_state": dict}                      │
    └───────────────────────────────────────────────┘
```

### 1-1. STFT Parameters (paper defaults)

| Name | Value | Meaning |
|---|---|---|
| `stft_chunk_size` | 96 | 6 ms hop @ 16 kHz |
| `stft_back_pad` | 96 | 6 ms lookback |
| `stft_pad_size` | 64 | 4 ms lookahead |
| `nfft` | 256 (= 96+96+64) | window length |
| `nfreqs` | 129 | nfft/2 + 1 |
| Algorithmic latency | 10 ms | chunk + lookahead = 6 + 4 |

`Net.__init__:63-72` constructs the analysis/synthesis (rect) windows from these values and uses `get_perfect_synthesis_window()` to produce a perfect-reconstruction window.

### 1-2. Forward Shape Evolution

```
mixture (B, 2, 80000)
  │
  │ mod_pad + back/fore pad
  ▼
(B, 2, 80000+pad)
  │ extract_features (torch.stft + view_as_real)
  ▼
(B, 2*2, T, F)  =  (B, 4, T, 129)   ← real/imag split, M=2 channels
  │ MultiFiLMGuidedTFNet (FiLM × 6 + GridNetBlock × 6)
  ▼
(B, S, T, 2F) = (B, 5, T, 258)
  │ synthesis (iFFT + overlap-add)
  ▼
(B, 5, 80000)
```

Here `T = (80000 + 96 + 64) // 96 + 1 ≈ 836` frames.

---

## 2. MultiFiLMGuidedTFNet — Separator Body

`src/tse/multiflim_guided_tfnet.py:20`

```
input: (B, 2M=4, T, F=129) + embedding (B, 20)
       │
       ▼
  ┌──────────────────────────────────────────────────────┐
  │ Conv2d(in=4, out=32, kernel=3×3, padding=(0,1))      │
  │   → (B, 32, T, 129)                                  │
  └──────────────────────────────────────────────────────┘
       │
       │  (use_first_ln=False, this model does not use LN)
       │
       ▼
  ┌──────────────────────────────────────────────────────┐
  │ for ii in range(6):  # n_layers = 6                  │
  │                                                      │
  │   ┌──────────────────────────────────────┐           │
  │   │ FiLM[ii]                             │           │
  │   │   src/tse/film.py:4                  │           │
  │   │   FiLM(x, emb) = x * a(emb) + b(emb) │           │
  │   │   a, b: Linear(20 → 32)              │           │
  │   └──────────────────────────────────────┘           │
  │           │                                          │
  │           ▼                                          │
  │   ┌──────────────────────────────────────┐           │
  │   │ GridNetBlock[ii]                     │           │
  │   │   src/tse/gridnet_block.py:6         │           │
  │   │   ├ intra-frame (freq) LSTM (bidir)  │           │
  │   │   │   - LayerNorm(32)                │           │
  │   │   │   - LSTM(32→64, bidir=True)      │           │
  │   │   │   - Linear(128→32)               │           │
  │   │   │   - residual                     │           │
  │   │   └ inter-frame (time) LSTM (causal) │           │
  │   │       - LayerNorm(32)                │           │
  │   │       - LSTM(32→64, bidir=False)     │           │
  │   │       - Linear(64→32)                │           │
  │   │       - residual                     │           │
  │   └──────────────────────────────────────┘           │
  └──────────────────────────────────────────────────────┘
       │
       ▼
  ┌──────────────────────────────────────────────────────┐
  │ ConvTranspose2d(in=32, out=5×2=10, kernel=3×3)       │
  │   → (B, 10, T, 129)                                  │
  │ reshape → (B, 5, T, 258)                             │
  └──────────────────────────────────────────────────────┘
       │
       ▼
   (B, 5, T, 258)
```

### 2-1. FiLM Preset

When `film_params: {film_preset: "all"}`, `multiflim_guided_tfnet.py:114-119` applies the following mapping.

| preset | applied positions |
|---|---|
| `"all"` | `[0, 1, 2, 3, 4, 5]` (in front of every block) |
| `"all_except_first"` | `[1, 2, 3, 4, 5]` |
| `"first"` | `[0]` |

> The TFGridNet Large reference here uses **`"all"`**.

### 2-2. Embedding Handling

When `embedding_params.embedding_type == ""` or `embedding_dim == 0`, no embedding layer is used and **the 20-dim label_vector is passed directly as `emb`** (`multiflim_guided_tfnet.py:65-68`). This is the case for the present config.

---

## 3. GridNetBlock — intra/inter LSTM

`src/tse/gridnet_block.py:6`

The input arrives as `(B, T, Q=129, C=32)` and is processed in two stages.

```
input (B, T, Q, C)
  │
  │  ── 1) intra-frame (process the frequency axis within each time frame) ──
  │     reshape → (B*T, Q, C)
  │     LayerNorm (C)
  │     LSTM(in=C, hid=H=64, bidir=True)
  │     Linear(2H → C)
  │     reshape → (B, T, Q, C)
  │     + residual
  ▼
intermediate (B, T, Q, C)
  │
  │  ── 2) inter-frame (process the time axis within each frequency bin) ──
  │     LayerNorm (C)
  │     transpose → (B*Q, T, C)
  │     LSTM(in=C, hid=H=64, bidir=False)   ← causal!
  │     Linear(H → C)
  │     reshape → (B, T, Q, C)
  │     + residual
  ▼
output (B, T, Q, C)
```

Key point: **the inter-frame LSTM is unidirectional (causal)** — enabling real-time/streaming processing.

### 3-1. Per-Block Parameter Distribution

A single GridNetBlock has roughly 81,600 parameters, so 6 blocks total ≈ 489,600.
The bulk lives in the LSTM weights (`intra_seq2seq`, `inter_rnn`).

```
Whole model (501,738 params)
├ tfgridnet.conv         :   1,184  (0.2%)
├ tfgridnet.film_layers  :   8,064  (1.6%)   ← 6 × FiLM(20→32, a/b)
├ tfgridnet.blocks       : 489,600  (97.6%)  ← 6 × GridNetBlock
└ tfgridnet.deconv       :   2,890  (0.6%)
```

---

## 4. Inference Branching (`src/tse/eval.py`)

```
inputs: mixture, label_vector
                │
                ▼
          ┌──────────────────────────────┐
          │ model.nO == 1 AND            │
          │ label_vector.sum() > 1?      │
          └──────────────┬───────────────┘
            yes          │           no
        ┌────────────────┘           └─────────────────┐
        ▼                                              ▼
┌───────────────────────────┐         ┌─────────────────────────────────┐
│ per-label loop            │         │ single forward (multi-hot)      │
│ for i in active_classes:  │         │   model({"mixture":x,           │
│   one-hot[i] = 1          │         │           "embedding":lv})      │
│   model(one-hot)          │         │   → (B, nO, T)                  │
│ stack → (1, N_fg, T)      │         └─────────────────────────────────┘
└───────────────────────────┘
```

This TFGridNet Large variant has `nO=5` (multi-output), so it takes the **single forward** path. The `nO=1` variants (e.g. the 1-out version of `orange_pi`) take the per-label loop path.

---

## 5. Loss Function

`src/tse/loss.py` — `MultiResoFuseLoss`:

```
loss = MultiResolutionSTFTLoss(auraloss) + l1_ratio * L1Loss
```

`l1_ratio = 10` (`configs/tse/orange_pi.yaml:51`).

Next document: [SED pipeline →](./sed-pipeline.md)
