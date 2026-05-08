# 04. 학습 GPU 리소스 산정 — TFGridNet Large

[← README 로](../README.md)

본 문서는 **TFGridNet Large** (`configs/tse/orange_pi.yaml`) 학습에 필요한 GPU VRAM과 시간을 **실측 + 외삽** 으로 산정합니다.

---

## 🚨 핵심 결론 (TL;DR)

| 항목 | 값 |
|---|---|
| 모델 파라미터 | 0.502 M (매우 작음) |
| **B=8 FP32 학습 VRAM** | **~34 GB** (예상) |
| **B=8 AMP FP16 학습 VRAM** | **~18 GB** (예상) |
| 권장 GPU (B=8 FP32) | **L40S 48GB × 2 (DDP)** 또는 A100 40GB 단일 (빡빡) |
| Colab 권장 | Pro+ A100 40GB (B=8 가능), L4 24GB (AMP 필요) |
| Colab 무료 (T4 16GB) | **불가** (B=2까지만, 학습 의미 없음) |

> 🔑 **왜 0.5M 모델이 30GB+ 를 먹나?**
> TFGridNet 의 LSTM 들이 backward 를 위해 **모든 시간 step 의 hidden state를 저장**합니다. intra-frame LSTM 은 effective batch `B×T` (= 8×836 = 6,688) 으로 동작 → activation 메모리가 모델 크기의 **수백 배**.

---

## 1. 실측 결과 (RTX 4070 Ti SUPER, 16 GB, FP32)

본 측정은 `MultiResoFuseLoss` 학습 forward + backward + optimizer.step 한 사이클 peak 메모리.

### 1-1. VRAM 사용량 (실측)

| Batch | FP32 + MR-STFT | AMP FP16 + MR-STFT | 비고 |
|---|---|---|---|
| 1 | **4.23 GB** ✅ 실측 | **2.22 GB** ✅ 실측 | |
| 2 | **8.69 GB** ✅ 실측 | **4.54 GB** ✅ 실측 | |
| 3 | OOM (16GB) | — | |
| 4 | (~17.4 GB 추정) | **9.04 GB** ✅ 실측 | |
| 8 | **~34.8 GB 추정** | ~18.1 GB 추정 | **사용자 보고: L40S(48GB) 1대 부족** |
| 16 | ~69.6 GB 추정 | ~36.2 GB 추정 | |

> 외삽은 sample 당 메모리가 거의 선형(~4.3 GB/sample FP32, ~2.3 GB/sample AMP) 이어서 신뢰도 높음. 다만 framework overhead, NCCL/DDP buffer, MR-STFT 의 multi-resolution 임시 텐서로 **+1~2 GB 마진** 필요.

### 1-2. Step time (실측)

| 설정 | step time | throughput |
|---|---|---|
| FP32 B=2 | **139.7 ms** | 14.3 samples/s |
| AMP B=4 | **203.9 ms** | 19.6 samples/s |

- GPU: RTX 4070 Ti SUPER (FP32 peak 44.1 TFLOPS, FP16 peak 88.2 TFLOPS)
- LSTM-heavy 모델이라 SM 활용도 낮음 (peak 의 ~3% 활용 — cuDNN LSTM 의 한계)
- AMP 가 step time 면에서는 약간 느림 — 하지만 **VRAM 절감으로 batch 를 키울 수 있어 sample/s 는 높음**

---

## 2. 메모리가 큰 이유 — TFGridNet 의 LSTM activation

```
입력 (B, 2, 80000) → STFT → (B, 4, T=836, F=129) → reshape → (B, T, F, D=32)
                                                                       │
6 × GridNetBlock 각각:                                                  │
  ├ intra-frame LSTM:                                                  │
  │   reshape → (B*T, F, D) = (8*836, 129, 32)                         │
  │   bidir LSTM(C=32 → H=64) → 모든 step hidden state 저장 (backward)  │
  │   activation: ~1.0 GB per block per batch element                  │
  │                                                                    │
  ├ inter-frame LSTM (causal):                                         │
  │   transpose → (B*F, T, D) = (8*129, 836, 32)                       │
  │   LSTM(C=32 → H=64) → 모든 step hidden state 저장                   │
  │   activation: ~0.8 GB per block per batch element                  │
  │                                                                    │
  └ ───────────                                                        │
       per-block activation total: ~1.8 GB × 6 blocks = ~10.8 GB       │
       + STFT + Conv2d + FiLM + autograd graph overhead                │
                                                                       │
                                       Total: ~25 GB (FP32, B=8) 추정  
```

**핵심**: intra-frame LSTM 의 effective batch 가 `B*T` = `8*836` = 6,688 입니다. RNN 은 backward 를 위해 **모든 time step 의 hidden + cell state** 를 저장해야 하므로 activation 이 폭발.

---

## 3. Colab GPU 적합성 매트릭스

| GPU | VRAM | FP32 TFLOPS | TF32 | FP16 | B=8 FP32 | B=8 AMP | B=4 AMP | B=2 FP32 |
|---|---|---|---|---|---|---|---|---|
| **T4** (Free) | 16 GB | 8.1 | — | 65.1 | ❌ | ❌ | ✅ (~9 GB) | ✅ (8.7 GB) |
| **V100** (Pro) | 16 GB | 14.0 | — | 112 | ❌ | ❌ | ✅ | ✅ |
| **L4** (Pro+) | 24 GB | 30.3 | 121 | 242 | ❌ | ✅ (~18 GB) | ✅ | ✅ |
| **A100 40 GB** (Pro+) | 40 GB | 19.5 | 156 | 312 | ⚠️ 빡빡 (~35 GB) | ✅ | ✅ | ✅ |
| **L40S 48 GB** (참고) | 48 GB | 91.6 | 183 | 362 | ⚠️ 마진 부족 | ✅ | ✅ | ✅ |
| **L40S × 2 DDP** (사용자 환경) | 96 GB | × 2 | × 2 | × 2 | ✅ B=8 분산 | ✅ B=16 분산 | ✅ | ✅ |
| **A100 80 GB** | 80 GB | 19.5 | 156 | 312 | ✅ (~35 GB) | ✅ B=16 | ✅ | ✅ |

> ⚠️ "빡빡" / "마진 부족" 표시는 framework overhead + DataLoader pin memory + NCCL 등을 더하면 OOM 위험이 있다는 뜻. **사용자 보고처럼 L40S 1 대로는 부족, 2 대 DDP 가 안전.**

### 3-1. Colab 무료 (T4 16 GB)

```
❌ 학습 권장 안 함.
   - B=2 가 최대 (8.7 GB) — 논문 batch=8 의 1/4 수준
   - AMP 로도 B=4 가 최대
   - 12 h 세션 timeout → 200 epoch 불가
   - 디스크 한계 (~100 GB) → 데이터셋(110 GB) 부분적
권장 용도: mini dataset (~250 MB) 으로 inference 검증만
```

### 3-2. Colab Pro (V100 / T4 / L4 가변)

```
🟡 학습 가능하나 batch_size 축소 필요.
   - V100 16 GB: T4 와 동일 한계 (B=2 FP32, B=4 AMP)
   - L4 24 GB:    B=4 FP32, B=8 AMP 가능
   - 24 h 세션 → 200 epoch 분할 학습 (수일 소요)
권장 batch_size: AMP B=4 또는 B=8 (L4 일 때)
```

### 3-3. Colab Pro+ (A100 40 GB)

```
✅ 권장 환경.
   - B=8 FP32 가능 (~35 GB, 마진 5 GB)
   - B=16 AMP 가능 (~36 GB)
   - TF32 자동 활성화로 matmul 가속 (소폭)
   - 24 h 세션 한계는 여전 → 200 epoch 분할 필요
   - torch.compile() 추가 가속 가능
```

---

## 4. 200 epoch wall-clock 추정

samples_per_epoch = 20,000, batch_size = 8 → 2,500 steps/epoch, 총 500,000 steps.

### 4-1. 실측 기반 step time 비례 추정

기준: RTX 4070 Ti Super 실측 (FP32 B=2 = 140 ms, AMP B=4 = 204 ms). FP32 TFLOPS 로 비례.

| GPU | FP32 TFLOPS | step time (FP32 B=8 추정) | throughput | 1 epoch | 200 epoch |
|---|---|---|---|---|---|
| 4070 Ti Super (실측) | 44.1 | 560 ms (B=8 외삽) | 14.3 s/s | 23 min | **77 hours (3.2 d)** |
| T4 (B=8 OOM) | 8.1 | — (불가) | — | — | — |
| V100 (B=8 OOM) | 14.0 | — | — | — | — |
| L4 24 GB (B=8 OOM) | 30.3 | — | — | — | — |
| A100 40 GB | 19.5 (FP32) | ~1,200 ms | 6.7 s/s | ~50 min | **~165 hours (7 d)** |
| A100 + TF32 | 156 (matmul) | ~400 ms 추정 | ~20 s/s | ~17 min | **~55 hours (2.3 d)** |
| L40S × 2 DDP | 91.6 × 2 | ~270 ms (effective) | ~30 s/s | ~11 min | **~37 hours (1.5 d)** |

> ⚠️ 위 추정은 **GPU compute 만** 고려. 실제로는 **데이터 로딩 (HRTF + LUFS) 병목** 이 추가됨.

### 4-2. 데이터 로딩 병목 영향

`SoundscapeDataset.__getitem__` 한 sample 처리 시간 (CPU):

- 3-9 audio 파일 read + resample → 50-200 ms
- LUFS normalize × 3-9 (각 100 iter) → 200-500 ms
- HRTF convolution × 3-9 → 100-300 ms
- 합계: **~500-1,000 ms / sample**

`num_workers=8` 기준 batch=8 채우는 시간 ≈ 0.5-1.0 s. **GPU step 보다 느림** → 데이터 로딩이 bound.

| 환경 | step time (compute) | step time (data 포함, 추정) |
|---|---|---|
| Colab vCPU 2 (num_workers 효과 제한) | — | **1.5-3.0 s/step** |
| Klone 워크스테이션 (CPU 16+, fast SSD) | — | 0.7-1.2 s/step |
| L40S × 2 DDP, 빠른 NVMe | 270 ms | 0.5-0.7 s/step |

→ Colab Pro+ A100 40GB 에서도 **현실적으로 200 epoch ≈ 5-10 일** 으로 봐야 함.

### 4-3. 가속 옵션 적용 시 (참고)

| 기법 | 메모리 절감 | 시간 절감 | 적용 위치 |
|---|---|---|---|
| AMP (FP16) | × 0.5 | × 1.0-1.2 | `pl.Trainer(precision="16-mixed")` |
| BF16 | × 0.5 | × 1.0-1.2 | `precision="bf16-mixed"` (A100/L40S) |
| `torch.compile()` | — | × 1.1-1.3 | `model = torch.compile(model)` |
| **mixture pre-synthesis** | — | **× 5-10** | mixture 를 디스크에 사전 합성 (현재 `SoundscapeDataset` 에는 미구현 — 레거시 `MisophoniaDataset.save_dataset()` 만 존재) |
| DDP (multi-GPU) | × 1/N | × ~0.6/N | `devices=N, strategy="ddp"` |

> **데이터 로딩 병목이 가장 큰 적**. mixture 를 사전 합성해 디스크에 저장하면 학습 시간이 5-10 배 단축됩니다 (단, on-the-fly augmentation 손실).

---

## 5. 권장 환경별 batch_size 가이드

| 환경 | precision | batch_size | 비고 |
|---|---|---|---|
| Colab Free (T4 16GB) | AMP | 4 | 200 epoch 비현실적; 검증용 |
| Colab Pro (V100 16GB) | AMP | 4 | 200 epoch 분할 학습 가능 |
| Colab Pro (L4 24GB) | AMP | 8 | 추천 |
| Colab Pro+ (A100 40GB) | FP32 | 8 | 논문 기본 |
| Colab Pro+ (A100 40GB) | AMP | 16 | throughput ↑ |
| L40S × 1 (48GB) | AMP | 8 | FP32 는 마진 부족 |
| L40S × 2 DDP (96GB) | FP32 | 8 (effective 16) | 사용자 검증 환경 |
| A100 80GB × 1 | FP32 | 16 | 권장 |

---

## 6. 직접 측정 스니펫

학습 시작 전 자기 환경에서 한 번 측정해 보길 권장. 측정 도구:

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

## 7. 측정 환경 명세 (재현성)

| 항목 | 값 |
|---|---|
| GPU | NVIDIA RTX 4070 Ti SUPER 16 GB |
| Driver | 550.54.14 (CUDA 12.4) |
| PyTorch | 2.6.0+cu124 |
| OS | Linux 5.15 |
| Python | 3.11 |
| 가상환경 | `uv venv .venv-mem-test` (uv 0.9.3) |
| 측정 cmd | 위의 스니펫 |
| 모델 config | `configs/tse/orange_pi.yaml` (D=32, H=64, B=6, 5-out) |
| 입력 길이 | 5 s @ 16 kHz (80,000 samples) |

다음: [20-class + noise 노트북 →](../04-samples/sound-samples.ipynb)
