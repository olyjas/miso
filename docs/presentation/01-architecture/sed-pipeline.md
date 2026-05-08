# 01-3. SED Pipeline Detail

[← Back to overview](./overview.md) · [← TSE detail](./tse-pipeline.md)

---

## 1. Model: Fine-tuned AST

`src/sed/ast_hf.py:25` — the `ASTHuggingFace` class wraps HuggingFace's `ASTForAudioClassification`.

```
HuggingFace model: MIT/ast-finetuned-audioset-10-10-0.4593
                   └ base: 86 M params, AudioSet 527-class pretrained
                   └ Fine-tuning: head replaced with 20-class (ignore_mismatched_sizes=True)
                   └ Checkpoint used by this repo: ooshyun/sound_event_detection
```

---

## 2. Forward Flow

```
inputs = {"mixture": (B, [M], T)}    ← mono-downmix when M channels
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
    "output":     logits,     ← (B, 20)  BCE loss at training time
    "scores":     scores,     ← (B, 20)  threshold applied at inference
    "embeddings": embeddings, ← (B, 768) CLS token
}
```

> **Important**: at inference you must use `outputs["scores"]` (softmax).
> Using `sigmoid(logits)` shifts the score distribution, so the per-class thresholds tuned on the val set no longer match and recall drops sharply.

---

## 3. Class Mapping

The model emits 20 classes. Class definitions live in `data/class_map.yaml`.

```
20 foreground classes (alphabetical):
  alarm_clock, baby_cry, birds_chirping, car_horn, cat,
  cock_a_doodle_doo, computer_typing, cricket, dog, door_knock,
  glass_breaking, gunshot, hammer, music, ocean,
  singing, siren, speech, thunderstorm, toilet_flush
```

Each class is expanded along its **subtree** in the AudioSet ontology to gather training data.
For example, `dog` collects `Dog`, `Bark`, `Yip`, `Howl`, `Bow-wow`, `Growling`, ... all into the dog folder.

---

## 4. Threshold Calibration at Evaluation Time

Core flow in `src/sed/eval.py`:

```
        ┌────────────────────────────────────┐
        │ 1. Build val dataset               │
        │    (when --find_thresholds)        │
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

### 4-1. Threshold Options

| Option | Behavior |
|---|---|
| `--find_thresholds` | Tune per-class F1-max thresholds on val and apply to test (paper method) |
| `--thresholds <json>` | Use a precomputed threshold JSON (`configs/sed/optimal_thresholds.json`) |
| Default | Fixed threshold of 0.5 |

---

## 5. Training Configuration — `configs/sed/ast_finetune.yaml`

| Item | Value | Note |
|---|---|---|
| Base model | `MIT/ast-finetuned-audioset-10-10-0.4593` | HuggingFace |
| num_classes | 20 | head replaced |
| freeze_encoder | `true` | encoder frozen, only head trained |
| sample_rate | 16000 | mono |
| batch_size | 32 | recommended |
| max_epochs | 80 | |
| encoder_lr | 1e-5 | (unused by default since freeze=True) |
| head_lr | 1e-3 | |
| weight_decay | 0.01 | |
| scheduler | CosineAnnealingWarmRestarts | T_0=10 |
| loss | BCEWithLogitsLoss | `pos_weight: "auto"` |
| primary_metric | mAP | |

> **Note**: BCE loss is applied to the logits, but **inference uses softmax(logits)** (a different objective than Cross-Entropy). It looks like a train/inference mismatch, but the model is intentionally designed to support both multi-label training and multi-class top-1 inference.

---

## 6. Baseline Models

Defined in `src/sed/model.py:_MODEL_NAME_MAP`:

| Name | Description |
|---|---|
| `finetuned_ast` | **This repo's model** (paper Table 4) — 20-class fine-tuned |
| `ast_pretrained` | Original MIT 527-class — softmax then mapped to 20 classes |
| `yamnet` | Google YAMNet (TF SavedModel) — 521-class baseline |

Next: [Preprocessing pipeline →](../02-preprocessing/overview.md)
