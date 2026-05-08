# 01-2. TSE 파이프라인 상세 (TFGridNet Large)

기준 모델: `orange_pi` config (`configs/tse/orange_pi.yaml`)

[← 개요로 돌아가기](./overview.md)

---

## 1. 최상위 흐름 — `Net.forward`

`Net` (`src/tse/net.py:31`) 은 STFT 프론트엔드 + 분리기(MultiFiLMGuidedTFNet) + iSTFT 백엔드를 묶어주는 래퍼입니다.

```
inputs = {"mixture": (B, 2, 80000), "embedding": (B, 20)}
            │
            ▼
    ┌───────────────────────────────────────────────┐
    │ Net.forward(inputs, input_state, pad)         │
    │   src/tse/net.py:216                          │
    │                                               │
    │   1. init_buffers(B, device) (state 없으면)   │
    │   2. predict(x, embedding, state, pad)        │
    │      ├─ mod_pad: chunk_size 정렬 + 양옆 패딩 │
    │      ├─ extract_features: STFT               │
    │      ├─ tfgridnet(x, emb, state)             │
    │      └─ synthesis: iFFT + overlap-add        │
    │                                               │
    │   returns:                                    │
    │     {"output": (B, 5, 80000),                 │
    │      "next_state": dict}                      │
    └───────────────────────────────────────────────┘
```

### 1-1. STFT 파라미터 (논문 기본값)

| 이름 | 값 | 의미 |
|---|---|---|
| `stft_chunk_size` | 96 | 6 ms hop @ 16 kHz |
| `stft_back_pad` | 96 | 6 ms lookback |
| `stft_pad_size` | 64 | 4 ms lookahead |
| `nfft` | 256 (= 96+96+64) | 윈도우 길이 |
| `nfreqs` | 129 | nfft/2 + 1 |
| 알고리즘 latency | 10 ms | chunk + lookahead = 6 + 4 |

`Net.__init__:63-72` 에서 위 값으로 분석/합성 윈도우(rect)를 만들고, `get_perfect_synthesis_window()` 로 perfect reconstruction 윈도우를 생성합니다.

### 1-2. Forward 데이터 셰이프 변화

```
mixture (B, 2, 80000)
  │
  │ mod_pad + back/fore pad
  ▼
(B, 2, 80000+pad)
  │ extract_features (torch.stft + view_as_real)
  ▼
(B, 2*2, T, F)  =  (B, 4, T, 129)   ← real+imag 분리, M=2 채널
  │ MultiFiLMGuidedTFNet (FiLM × 6 + GridNetBlock × 6)
  ▼
(B, S, T, 2F) = (B, 5, T, 258)
  │ synthesis (iFFT + overlap-add)
  ▼
(B, 5, 80000)
```

여기서 `T = (80000 + 96 + 64) // 96 + 1 ≈ 836` 프레임.

---

## 2. MultiFiLMGuidedTFNet — 분리기 본체

`src/tse/multiflim_guided_tfnet.py:20`

```
입력: (B, 2M=4, T, F=129) + embedding (B, 20)
       │
       ▼
  ┌──────────────────────────────────────────────────────┐
  │ Conv2d(in=4, out=32, kernel=3×3, padding=(0,1))      │
  │   → (B, 32, T, 129)                                  │
  └──────────────────────────────────────────────────────┘
       │
       │  (use_first_ln=False, 본 모델은 LN 미사용)
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

### 2-1. FiLM preset

`film_params: {film_preset: "all"}` 이면 `multiflim_guided_tfnet.py:114-119` 에서 다음 매핑 적용.

| preset | 적용 위치 |
|---|---|
| `"all"` | `[0, 1, 2, 3, 4, 5]` (모든 블록 앞) |
| `"all_except_first"` | `[1, 2, 3, 4, 5]` |
| `"first"` | `[0]` |

> 본 TFGridNet Large 기준은 **`"all"`**.

### 2-2. embedding 처리

`embedding_params.embedding_type == ""` 또는 `embedding_dim == 0` 이면 임베딩 레이어 없이 **label_vector(20-dim)을 그대로 emb 으로 사용**합니다 (`multiflim_guided_tfnet.py:65-68`). 본 config 가 이 경우입니다.

---

## 3. GridNetBlock — intra/inter LSTM

`src/tse/gridnet_block.py:6`

`(B, T, Q=129, C=32)` 형태로 들어와 두 단계 처리.

```
입력 (B, T, Q, C)
  │
  │  ── 1) intra-frame (각 시간 프레임 내 frequency 축 처리) ──
  │     reshape → (B*T, Q, C)
  │     LayerNorm (C)
  │     LSTM(in=C, hid=H=64, bidir=True)
  │     Linear(2H → C)
  │     reshape → (B, T, Q, C)
  │     + residual
  ▼
중간 (B, T, Q, C)
  │
  │  ── 2) inter-frame (각 frequency bin 내 time 축 처리) ──
  │     LayerNorm (C)
  │     transpose → (B*Q, T, C)
  │     LSTM(in=C, hid=H=64, bidir=False)   ← causal!
  │     Linear(H → C)
  │     reshape → (B, T, Q, C)
  │     + residual
  ▼
출력 (B, T, Q, C)
```

핵심: **inter-frame LSTM은 단방향(causal)** — 실시간/스트리밍 처리 가능.

### 3-1. 블록당 파라미터 분포

블록 1개(GridNetBlock) 파라미터 ≈ 81,600. × 6 블록 = 489,600.
대부분 LSTM 가중치(`intra_seq2seq`, `inter_rnn`)에 집중.

```
모델 전체 (501,738 params)
├ tfgridnet.conv         :   1,184  (0.2%)
├ tfgridnet.film_layers  :   8,064  (1.6%)   ← 6 × FiLM(20→32, a/b)
├ tfgridnet.blocks       : 489,600  (97.6%)  ← 6 × GridNetBlock
└ tfgridnet.deconv       :   2,890  (0.6%)
```

---

## 4. 추론 시 분기 (`src/tse/eval.py`)

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

본 TFGridNet Large 는 `nO=5` (multi-output) 이므로 **single forward** 경로를 사용합니다. `nO=1` 변형(예: `orange_pi` 의 1-out 버전)은 per-label loop 사용.

---

## 5. 손실 함수

`src/tse/loss.py` — `MultiResoFuseLoss`:

```
loss = MultiResolutionSTFTLoss(auraloss) + l1_ratio * L1Loss
```

`l1_ratio = 10` (`configs/tse/orange_pi.yaml:51`).

다음 문서: [SED 파이프라인 →](./sed-pipeline.md)
