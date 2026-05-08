# 01-3. SED 파이프라인 상세

[← 개요로 돌아가기](./overview.md) · [← TSE 상세](./tse-pipeline.md)

---

## 1. 모델: Fine-tuned AST

`src/sed/ast_hf.py:25` — `ASTHuggingFace` 클래스가 HuggingFace `ASTForAudioClassification`을 래핑합니다.

```
HuggingFace 모델: MIT/ast-finetuned-audioset-10-10-0.4593
                   └ 베이스: 86 M params, AudioSet 527-class pretrained
                   └ Fine-tuning: 20-class 헤드로 교체 (ignore_mismatched_sizes=True)
                   └ 본 레포 사용 체크포인트: ooshyun/sound_event_detection
```

---

## 2. Forward 흐름

```
inputs = {"mixture": (B, [M], T)}    ← M 채널이면 mono-downmix
          │
          ▼
┌──────────────────────────────────────────────────────┐
│ ASTHuggingFace.forward(inputs)                       │
│   src/sed/ast_hf.py:218                              │
│                                                      │
│   1. mono-downmix:  mixture.mean(dim=1) if M>1       │
│   2. peak-normalize: x / max(|x|)                    │
│   3. predict(waveform)                               │
└──────────────────┬───────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────┐
│ ASTHuggingFace.predict(waveform)                     │
│   src/sed/ast_hf.py:157                              │
│                                                      │
│   1. AutoFeatureExtractor → log-mel spectrogram      │
│   2. ast_model(input_values, output_hidden_states)   │
│   3. logits = outputs.logits           (B, 20)       │
│   4. scores = softmax(logits, dim=-1)  (B, 20)       │
│   5. embeddings = hidden_states[-1][:,0,:]  (B, 768) │
│                                                      │
│   returns (logits, scores, embeddings)               │
└──────────────────┬───────────────────────────────────┘
                   │
                   ▼
output = {
    "output":     logits,     ← (B, 20)  학습 시 BCE 손실
    "scores":     scores,     ← (B, 20)  추론 시 threshold 적용
    "embeddings": embeddings, ← (B, 768) CLS token
}
```

> **중요**: 추론 시 반드시 `outputs["scores"]` (softmax) 를 사용해야 합니다.
> `sigmoid(logits)` 를 쓰면 score 분포가 달라져서 val에서 찾은 per-class threshold가 어긋나고 recall 이 크게 떨어집니다.

---

## 3. 클래스 매핑

본 모델은 20-class 출력. 클래스 정의: `data/class_map.yaml`.

```
20 foreground classes (alphabetical):
  alarm_clock, baby_cry, birds_chirping, car_horn, cat,
  cock_a_doodle_doo, computer_typing, cricket, dog, door_knock,
  glass_breaking, gunshot, hammer, music, ocean,
  singing, siren, speech, thunderstorm, toilet_flush
```

각 클래스는 AudioSet ontology 의 **subtree** 로 확장되어 학습 데이터가 모입니다.
예: `dog` → `Dog`, `Bark`, `Yip`, `Howl`, `Bow-wow`, `Growling`, ... 모두 dog 폴더에 모임.

---

## 4. 평가 시 threshold 보정

`src/sed/eval.py` 의 핵심 흐름:

```
        ┌────────────────────────────────────┐
        │ 1. Build val dataset               │
        │    (--find_thresholds 일 때)       │
        └────────────────┬───────────────────┘
                         ▼
        ┌────────────────────────────────────┐
        │ 2. Run val inference               │
        │    → scores (N_val, 20)            │
        │    → labels (N_val, 20)            │
        └────────────────┬───────────────────┘
                         ▼
        ┌────────────────────────────────────────────┐
        │ 3. _find_optimal_thresholds()              │
        │    sklearn.metrics.precision_recall_curve  │
        │    per-class F1-max threshold              │
        │    → thresholds (20,)                      │
        └────────────────┬───────────────────────────┘
                         ▼
        ┌────────────────────────────────────┐
        │ 4. Build test dataset              │
        │ 5. Run test inference              │
        │ 6. Apply thresholds → predictions  │
        │ 7. Compute Acc / P / R / F1 / mAP  │
        └────────────────────────────────────┘
```

### 4-1. threshold 옵션

| 옵션 | 동작 |
|---|---|
| `--find_thresholds` | val로 per-class F1-max threshold 찾고 test에 적용 (논문 방식) |
| `--thresholds <json>` | 사전 계산된 threshold JSON 사용 (`configs/sed/optimal_thresholds.json`) |
| 기본값 | 고정 threshold 0.5 |

---

## 5. 학습 설정 — `configs/sed/ast_finetune.yaml`

| 항목 | 값 | 비고 |
|---|---|---|
| 베이스 모델 | `MIT/ast-finetuned-audioset-10-10-0.4593` | HuggingFace |
| num_classes | 20 | head 교체 |
| freeze_encoder | `true` | encoder 동결, head 만 학습 |
| sample_rate | 16000 | mono |
| batch_size | 32 | 권장 |
| max_epochs | 80 | |
| encoder_lr | 1e-5 | (freeze=True 라 미사용 기본) |
| head_lr | 1e-3 | |
| weight_decay | 0.01 | |
| scheduler | CosineAnnealingWarmRestarts | T_0=10 |
| loss | BCEWithLogitsLoss | `pos_weight: "auto"` |
| primary_metric | mAP | |

> **주의**: 손실은 BCE 를 logits 에 적용하지만, **추론 시에는 softmax(logits)** 를 사용합니다 (Cross-Entropy 와 다른 목표). 학습/추론 mismatch 처럼 보이지만, 이 모델은 multi-label 학습 + multi-class top-1 추론 양쪽 모두 가능하도록 설계되어 있습니다.

---

## 6. 베이스라인 모델

`src/sed/model.py:_MODEL_NAME_MAP` 에 정의:

| 이름 | 설명 |
|---|---|
| `finetuned_ast` | **본 레포 모델** (논문 Table 4) — 20-class fine-tuned |
| `ast_pretrained` | MIT 원본 527-class — softmax 후 20 class 매핑 |
| `yamnet` | Google YAMNet (TF SavedModel) — 521-class baseline |

다음: [전처리 파이프라인 →](../02-preprocessing/overview.md)
