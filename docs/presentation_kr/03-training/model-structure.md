# 03-3. 모델 구조 (TFGridNet Large)

[← 학습 개요](./overview.md) · [← Trainer 구조](./trainer-structure.md)

본 문서는 학습 관점에서 본 모델 구조 요약입니다. 추론 분기·STFT 세부는 [01-2. TSE 파이프라인](../01-architecture/tse-pipeline.md) 에 더 자세합니다.

---

## 1. 학습 시 모델 호출 시퀀스

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
                            │ (참고: forward 는 inputs 에서
                            │  "mixture" 와 "embedding" 을 읽음.
                            │  학습 코드에서 label_vector → embedding
                            │  으로 키 매핑 또는 dataset 단에서 처리)
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

## 2. 클래스 계층

```
src.tse.net.Net                              ← STFT + 분리기 + iSTFT 래퍼
├ self.tfgridnet = MultiFiLMGuidedTFNet      ← 분리기 본체
│   ├ self.conv = nn.Conv2d(num_inputs=4, latent_dim=32, k=3x3)
│   ├ self.embedding = None                  ← embedding_dim=0 이므로 None
│   ├ self.film_layers = nn.ModuleDict({
│   │     "film_layer_0": FiLM(D=32, emb=20),
│   │     "film_layer_1": FiLM(D=32, emb=20),
│   │     ...
│   │     "film_layer_5": FiLM(D=32, emb=20),
│   │   })   ← film_preset="all" → 6 개 FiLM
│   ├ self.blocks = nn.ModuleList([
│   │     GridNetBlock(D=32, n_freqs=129, hidden=64),  × 6
│   │   ])
│   └ self.deconv = nn.ConvTranspose2d(D=32, n_srcs*2=10, k=3x3)
│
├ self.analysis_window  (rect, length nfft=256)
└ self.synthesis_window (perfect reconstruction window)
```

### 2-1. 파라미터 카운트 (실제 측정값)

| 모듈 | params | 비율 |
|---|---|---|
| `tfgridnet.conv` | 1,184 | 0.2% |
| `tfgridnet.film_layers` (6 × FiLM) | 8,064 | 1.6% |
| `tfgridnet.blocks` (6 × GridNetBlock) | **489,600** | **97.6%** |
| `tfgridnet.deconv` | 2,890 | 0.6% |
| **총계** | **501,738 (0.502 M)** | 100% |

> **체감 용량**: 0.5M params × 4 bytes (fp32) ≈ **2 MB** 모델 파일 — 매우 가벼움.

### 2-2. GridNetBlock 내부 (블록 1개)

```
GridNetBlock(latent_dim=32, n_freqs=129, hidden_channels=64)
  │
  ├ intra-frame branch (frequency 축 처리)
  │   ├ LayerNorm(32)                                    →   64 params
  │   ├ LSTM(in=32, hid=64, layers=1, bidir=True)        ← 49,920 params
  │   └ Linear(128 → 32)                                 →  4,128 params
  │   subtotal: ~54,000
  │
  └ inter-frame branch (time 축 처리, causal)
      ├ LayerNorm(32)                                    →     64 params
      ├ LSTM(in=32, hid=64, layers=1, bidir=False)       ← 24,960 params
      └ Linear(64 → 32)                                  →  2,080 params
      subtotal: ~27,000

블록 1개 총 ≈ 81,600 params. × 6 블록 = 489,600 params.
```

### 2-3. FiLM 1개

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

## 3. 손실 — `MultiResoFuseLoss`

`src/tse/loss.py:MultiResoFuseLoss`

```
loss = MultiResolutionSTFTLoss(auraloss) + l1_ratio * L1Loss
       │                                    │
       │ Multi-scale STFT loss              │ Time-domain L1 loss
       │ (인지적으로 의미 있는 spectral 거리) │
       │                                    │
       └─ 다중 해상도(다양한 nfft, hop) 평균  └─ l1_ratio=10 (config)
```

`MultiResolutionSTFTLoss` 의 기본 설정은 [auraloss 기본값](https://github.com/csteinmetz1/auraloss) 사용.

---

## 4. Optimizer / Scheduler

`src/tse/train.py:271-277` :

```python
# dotted-path 빌드 — 클래스/하이퍼파라미터 모두 yaml 에서 결정
optimizer = _import_attr(tc["optimizer_name"])(
    model.parameters(),
    **tc.get("optimizer_params", {}),
)
scheduler = _import_attr(tc["scheduler_name"])(
    optimizer,
    **tc.get("scheduler_params", {}),
)
```

`configs/tse/orange_pi.yaml:43-50` 의 키 매칭:

```yaml
optimizer_name: "torch.optim.AdamW"
optimizer_params: { lr: 0.001, weight_decay: 0.0 }
scheduler_name: "torch.optim.lr_scheduler.ReduceLROnPlateau"
scheduler_params: { factor: 0.5, patience: 5 }
```

- `monitor` 는 `val/loss` (config `checkpointing.monitor`).
- ReduceLROnPlateau 는 `val/loss` 가 5 epoch 동안 개선되지 않으면 lr × 0.5.
- gradient clipping: `grad_clip=1.0` (Lightning 의 `gradient_clip_val` 로 세팅됨; 본 레포 코드에선 `pl.Trainer(...)` 의 추가 kwargs 또는 `_extra_kwargs` 로 전달 시 활성).

---

## 5. 학습 루프에서 한 step

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
self.log("train/loss", loss)     ← Lightning 기본 logger 로 전송
```

**한 step 의 이론 compute**: ~111G MACs (forward 37G + backward ~74G), ≈ 0.22 TFLOP.

> ⚠️ **메모리는 가볍지 않습니다.** 파라미터(0.5 M)는 작지만 LSTM 의 backward activation 때문에 batch=8 FP32 학습 시 **~34 GB VRAM** 이 필요합니다.

다음: [wandb 사용 예제 →](./wandb-accelerator.md)
