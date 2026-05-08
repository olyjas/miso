# 02-3. Online Batch Creation — `SoundscapeDataset.__getitem__`

[← Preprocessing Overview](./overview.md) · [← A. Offline Split](./dataset-split.md)

During training, each batch is built by **on-the-fly binaural synthesis** for every sample. There is no precomputed mixture WAV on disk — a **different random scene** is generated each time.

Core code: `src/datasets/soundscape_dataset.py` (used by both `src/tse/train.py` and `src/sed/train.py`, plus the modern `src/tse/eval.py` test path; the legacy `MisophoniaDataset` is kept only for paper-reproduction bit-exactness).

---

## 1. How a Single Sample Is Built

```
__getitem__(idx)                  src/datasets/soundscape_dataset.py:421
  │
  │ 1. seed = idx + np.random.randint(1e6)  if split=="train" else seed = idx
  │    rng = np.random.RandomState(seed)                              :423
  │
  ▼
create_scene(rng)                                                      :313
  │
  ├─ 2. n_fg = rng.randint(num_fg_range)     # config: U[1, 5]         :320
  │     n_bg = rng.randint(num_bg_range)     # config: U[1, 3]
  │     n_noise = rng.randint(num_noise_range)  # config: U[1, 1]
  │
  ├─ 3. for _ in range(n_fg):                                          :334
  │       src_id  = rng.randint(0, len(fg_sounds))
  │       label   = fg_sounds[src_id]
  │       audio   = _get_random_snippet(fg_dir, label,                 :339
  │                                     rng, require_power=True)
  │         ├ pwr_threshold = self.pwr_threshold (= -40 dB)            :291
  │         ├ outer loop x 100, inner snippet loop x 10                :292-294
  │         ├ accept only snippets with pwr_dB > pwr_threshold         :301
  │         │   → reject "silent" segments with no events
  │         ├ if both loops exhaust, relax pwr_threshold by -1 dB      :304
  │         └ inside sample_snippet: sf.read + resample(16kHz)
  │       label_vector[src_id] = 1            # multi-hot              :346
  │
  ├─ 4. for n_bg sources: same logic, require_power=True               :356
  │     for n_noise:      require_power=False                          :369
  │                       ※ noise bypasses the power filter (urban
  │                         noise is naturally low-energy but valid)
  │
  ├─ 5. SNR / LUFS normalization                                       :375-391
  │       fg_snr = U[5, 15] dB   (config snr_range_fg)
  │       bg_snr = U[0, 10] dB   (config snr_range_bg)
  │       fg_lufs = ref_db + fg_snr  =  -50 + fg_snr
  │       bg_lufs = ref_db + bg_snr  =  -50 + bg_snr
  │       noise_lufs = ref_db        =  -50
  │       _normalize_to_lufs(audio, sr, target_lufs)                   :62
  │         ← iterative refinement via pyloudnorm
  │
  ├─ 6. Spatialise via HRTF                                            :396
  │       bi_srcs, bi_noise = self.hrtf_simulator.simulate(
  │           fg_audio + bg_audio, noise_audio, seed)
  │       (CIPICSimulator: src/datasets/hrtf.py)
  │
  ├─ 7. mixture = sum(bi_srcs) + bi_noise              # (2, 80000)    :399
  │
  ├─ 8. Fill in ground truth (gt) — sequentially [0:n_fg]              :405-413
  │       for i in range(n_fg):
  │           if task == "tse": gt[i] = mono(bi_srcs[i])
  │           else:             gt += binaural(bi_srcs[i])  # SED
  │       Channels [n_fg : num_output_channels] stay zero-padded
  │       (the metric adapter masks them out via num_fg_labels).
  │
  ▼
__getitem__ post-processing:
  │
  ├─ 9. AudioAugmentations (training only)                             :432-435
  │       perturbations.apply_random_augmentations(mixture, target, rng)
  │
  ├─ 10. peak normalize                                                :438-441
  │       if mixture.abs().max() > 1: mixture /= peak; target /= peak
  │
  ├─ 11. label padding (for default-collate compatibility)             :460-461
  │       fg_labels += ["None"] * (num_fg_max - n_fg)
  │
  ▼
return inputs, targets                                                 :469
```

---

## 2. Return Format

```python
# TSE (task="tse")
inputs = {
    "mixture":      torch.Tensor,  # (2, 80000)        binaural, float32
    "label_vector": torch.Tensor,  # (num_total_labels,) multi-hot, float32
    "embedding":    torch.Tensor,  # alias of label_vector — Net.forward consumes "embedding"
}

# SED (task="sed")
inputs = {
    "mixture": torch.Tensor,       # (2, 80000)
    "labels":  torch.Tensor,       # (num_total_labels,) multi-hot — BCE target
}

# Both tasks:
targets = {
    "target":         torch.Tensor,  # (num_output_channels, 80000) — TSE: per-source mono;
                                     #                                SED: summed binaural
    "fg_labels":      list[str],     # length num_fg_max, padded with "None"
    "num_fg_labels":  int,           # actual n_fg (≤ num_fg_max ≤ num_output_channels)
}
```

> **Key tensors**:
> - `mixture (2, 80000)` — training/inference input, 5-second binaural audio at 16 kHz
> - `label_vector / embedding (20,)` — FiLM conditioning input for TSE (multi-hot)
> - `target (num_out, 80000)` — TSE ground truth (per-source mono, channels [0:n_fg] only); for SED, `labels` (multi-hot) is the BCE target instead.
> - `num_fg_labels (B,)` — used by the validation metric adapter to mask out padded zero channels (see `src/tse/train.py:_build_metrics_fn`).

---

## 3. Batch Collation in the DataLoader

`src/tse/train.py:241` uses PyTorch's default `DataLoader` directly:

```python
train_loader = DataLoader(
    train_ds,
    batch_size=d.get("batch_size", 8),    # config: 8
    shuffle=True,
    num_workers=d.get("num_workers", 8),  # config: 8
    pin_memory=True,
    drop_last=True,
)
```

The default collator merges the dict-of-tensors into a batched dict:

```
Batch dict:
  inputs["mixture"]:        (8, 2, 80000)
  inputs["label_vector"]:   (8, 20)
  inputs["embedding"]:      (8, 20)
  targets["target"]:        (8, 5, 80000)    ← num_output_channels=5
  targets["fg_labels"]:     list[tuple[str, ...]]  (num_fg_max tuples of B strings)
  targets["num_fg_labels"]: torch.LongTensor (8,)
```

Non-tensor fields such as `fg_labels` (string lists) are passed through unchanged by the default collator.

SED uses a custom `_collate_fn` (`src/sed/train.py:64`) that stacks `mixture` and `labels` and returns `targets` as a tuple of dicts (since fg_labels list-of-tuples is not needed by the BCE loss).

---

## 4. Where the Heavy Work Happens (Performance Hotspots)

CPU cost breakdown of a single `__getitem__` call:

```
┌──────────────────────────────────────────────────────┐
│ 1. File I/O (open + read snippet)                    │
│    sf.SoundFile + read                       ~ 5-15% │
│                                                      │
│ 2. Resample to 16kHz                                 │
│    librosa.resample(kaiser_best) (CPU)       ~ 10-20%│
│    or torchaudio sinc_interp_hann            ~ 5-10% │
│                                                      │
│ 3. LUFS normalize  ← typically the largest cost      │
│    pyloudnorm.Meter().integrated_loudness            │
│    + iterative refinement                            │
│    fg/bg/noise each (3-9 calls total)        ~ 30-50%│
│                                                      │
│ 4. CIPIC HRTF convolution                            │
│    hrtf_simulator.simulate()                         │
│    binaural rendering per source             ~ 20-30%│
│                                                      │
│ 5. Audio augmentations (training only)               │
│    pitch / time / EQ etc.                     ~ 5-10%│
└──────────────────────────────────────────────────────┘
```

→ As a result, the training-throughput bottleneck is **CPU-side data loading** (the model itself is very lightweight). Tuning `num_workers`, `pin_memory`, and `prefetch_factor` matters more than GPU capability.

> **Note**: Unlike the legacy `MisophoniaDataset`, `SoundscapeDataset` does not pin BLAS threads internally. If thread contention surfaces under high `num_workers`, set `OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1` at the launcher level (or call `torch.set_num_threads(1)` in a `worker_init_fn`).

---

## 5. Deterministic vs Non-Deterministic Behavior

| split | seed policy | Effect |
|---|---|---|
| `train` | `seed = idx + np.random.randint(1e6)` | Different scene every epoch (acts as data augmentation) |
| `val` / `test` | `seed = idx` | Fixed scene independent of epoch → comparable metrics |

See `__getitem__:423-426`.

> One caveat: the `np.random.randint(1e6)` call in train mode draws from the worker's global numpy RNG. Unless numpy seeds are split per worker via PyTorch's `worker_init_fn`, multiple workers can end up sharing the same seed offset.

Next: [Training — Trainer / Model / wandb →](../03-training/overview.md)
