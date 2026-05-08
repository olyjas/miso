# 02-3. 온라인 batch 생성 — `SoundscapeDataset.__getitem__`

[← 전처리 개요](./overview.md) · [← A. 오프라인 split](./dataset-split.md)

학습 시 매 batch 는 sample 단위로 **on-the-fly binaural 합성** 으로 생성됩니다. 디스크에 미리 만들어 둔 mixture WAV 는 없고, 매번 **다른 random scene** 이 만들어집니다.

핵심 코드: `src/datasets/soundscape_dataset.py` (`src/tse/train.py`, `src/sed/train.py`, 및 최신 `src/tse/eval.py` 의 test 경로 모두 사용. 레거시 `MisophoniaDataset` 은 paper reproduction bit-exactness 용으로만 잔존).

---

## 1. 단일 sample 합성 흐름

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
  │         ├ outer 루프 100, inner snippet 루프 10                    :292-294
  │         ├ pwr_dB > pwr_threshold 인 snippet 만 accept              :301
  │         │   → 이벤트 없는 silent 구간 reject
  │         ├ 두 루프 모두 소진하면 pwr_threshold 를 -1 dB 완화         :304
  │         └ sample_snippet 내부: sf.read + resample(16kHz)
  │       label_vector[src_id] = 1            # multi-hot              :346
  │
  ├─ 4. n_bg sources 에도 같은 로직, require_power=True                :356
  │     n_noise 는 require_power=False                                 :369
  │                       ※ noise 는 power filter 우회 (도시 소음은
  │                         원래 작아도 valid 한 데이터)
  │
  ├─ 5. SNR / LUFS normalize                                           :375-391
  │       fg_snr = U[5, 15] dB   (config snr_range_fg)
  │       bg_snr = U[0, 10] dB   (config snr_range_bg)
  │       fg_lufs = ref_db + fg_snr  =  -50 + fg_snr
  │       bg_lufs = ref_db + bg_snr  =  -50 + bg_snr
  │       noise_lufs = ref_db        =  -50
  │       _normalize_to_lufs(audio, sr, target_lufs)                   :62
  │         ← pyloudnorm 반복 보정
  │
  ├─ 6. HRTF spatialise                                                :396
  │       bi_srcs, bi_noise = self.hrtf_simulator.simulate(
  │           fg_audio + bg_audio, noise_audio, seed)
  │       (CIPICSimulator: src/datasets/hrtf.py)
  │
  ├─ 7. mixture = sum(bi_srcs) + bi_noise              # (2, 80000)    :399
  │
  ├─ 8. ground truth (gt) 채우기 — 채널 [0:n_fg] 순차                  :405-413
  │       for i in range(n_fg):
  │           if task == "tse": gt[i] = mono(bi_srcs[i])
  │           else:             gt += binaural(bi_srcs[i])  # SED
  │       채널 [n_fg : num_output_channels] 는 zero-padded
  │       (metric adapter 가 num_fg_labels 로 마스킹).
  │
  ▼
__getitem__ 후처리:
  │
  ├─ 9. AudioAugmentations (train 전용)                                :432-435
  │       perturbations.apply_random_augmentations(mixture, target, rng)
  │
  ├─ 10. peak normalize                                                :438-441
  │       if mixture.abs().max() > 1: mixture /= peak; target /= peak
  │
  ├─ 11. label padding (default-collate 호환)                          :460-461
  │       fg_labels += ["None"] * (num_fg_max - n_fg)
  │
  ▼
return inputs, targets                                                 :469
```

---

## 2. 반환 포맷

```python
# TSE (task="tse")
inputs = {
    "mixture":      torch.Tensor,  # (2, 80000)        binaural, float32
    "label_vector": torch.Tensor,  # (num_total_labels,) multi-hot, float32
    "embedding":    torch.Tensor,  # label_vector 의 alias — Net.forward 가 "embedding" 키 사용
}

# SED (task="sed")
inputs = {
    "mixture": torch.Tensor,       # (2, 80000)
    "labels":  torch.Tensor,       # (num_total_labels,) multi-hot — BCE target
}

# 두 task 공통:
targets = {
    "target":         torch.Tensor,  # (num_output_channels, 80000) — TSE: per-source mono;
                                     #                                SED: 합산 binaural
    "fg_labels":      list[str],     # 길이 num_fg_max, "None" 으로 padding
    "num_fg_labels":  int,           # 실제 n_fg (≤ num_fg_max ≤ num_output_channels)
}
```

> **핵심 텐서**:
> - `mixture (2, 80000)` — 학습/추론 입력, 5초 binaural 16 kHz
> - `label_vector / embedding (20,)` — TSE FiLM conditioning 입력 (multi-hot)
> - `target (num_out, 80000)` — TSE GT (per-source mono, 채널 [0:n_fg] 만 채워짐). SED 는 별도로 `labels` (multi-hot) 가 BCE target.
> - `num_fg_labels (B,)` — validation metric adapter 가 padded zero 채널을 마스킹할 때 사용 (`src/tse/train.py:_build_metrics_fn`).

---

## 3. DataLoader 의 batch collation

`src/tse/train.py:241` 에서 PyTorch 기본 `DataLoader` 그대로 사용:

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

기본 collate 가 dict-of-tensors 를 batch dict 로 묶어줌:

```
Batch dict:
  inputs["mixture"]:        (8, 2, 80000)
  inputs["label_vector"]:   (8, 20)
  inputs["embedding"]:      (8, 20)
  targets["target"]:        (8, 5, 80000)    ← num_output_channels=5
  targets["fg_labels"]:     list[tuple[str, ...]]  (num_fg_max 개의 B-tuple)
  targets["num_fg_labels"]: torch.LongTensor (8,)
```

`fg_labels` 같은 string list 는 기본 collate 가 그대로 통과시킴.

SED 는 별도로 `_collate_fn` (`src/sed/train.py:64`) 을 사용 — `mixture` 와 `labels` 를 stack 하고 `targets` 는 dict 의 tuple 로 반환 (BCE loss 가 fg_labels 를 안 쓰기 때문).

---

## 4. 시간 비용이 큰 부분 (Performance Hotspots)

`__getitem__` 1회 처리 CPU 비용 분포:

```
┌──────────────────────────────────────────────────────┐
│ 1. File I/O (open + read snippet)                    │
│    sf.SoundFile + read                       ~ 5-15% │
│                                                      │
│ 2. Resample to 16kHz                                 │
│    librosa.resample(kaiser_best) (CPU)       ~ 10-20%│
│    or torchaudio sinc_interp_hann            ~ 5-10% │
│                                                      │
│ 3. LUFS normalize  ← 보통 가장 큰 비용                │
│    pyloudnorm.Meter().integrated_loudness            │
│    + 반복 보정                                        │
│    fg/bg/noise 각각 (총 3-9 회)              ~ 30-50%│
│                                                      │
│ 4. CIPIC HRTF convolution                            │
│    hrtf_simulator.simulate()                         │
│    source 별 binaural rendering              ~ 20-30%│
│                                                      │
│ 5. Audio augmentations (train 전용)                  │
│    pitch / time / EQ 등                       ~ 5-10%│
└──────────────────────────────────────────────────────┘
```

→ 결국 학습 throughput 의 병목은 **CPU-side data loading**. 모델 자체는 가볍기 때문에 GPU 보다 `num_workers`, `pin_memory`, `prefetch_factor` 튜닝이 더 효과적.

> **참고**: 레거시 `MisophoniaDataset` 과 달리 `SoundscapeDataset` 은 BLAS 스레드를 내부에서 고정하지 않습니다. `num_workers` 가 클 때 thread 경쟁이 보이면 launcher 단에서 `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1` 을 걸거나 `worker_init_fn` 에서 `torch.set_num_threads(1)` 호출하세요.

---

## 5. Deterministic vs Non-Deterministic

| split | seed 정책 | 효과 |
|---|---|---|
| `train` | `seed = idx + np.random.randint(1e6)` | epoch 마다 다른 scene (data augmentation 효과) |
| `val` / `test` | `seed = idx` | epoch 와 무관하게 고정된 scene → metric 비교 가능 |

`__getitem__:423-426` 참조.

> 한 가지 주의: train mode 의 `np.random.randint(1e6)` 는 worker 의 global numpy RNG 에서 뽑힙니다. PyTorch `worker_init_fn` 으로 worker 별 numpy seed 를 split 하지 않으면 여러 worker 가 같은 seed offset 을 공유할 수 있습니다.

다음: [학습 — Trainer / 모델 / wandb →](../03-training/overview.md)
