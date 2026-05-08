# 01-1. 시스템 아키텍처 — 개요

이 문서는 **TSE (Target Sound Extraction)** 와 **SED (Sound Event Detection)** 두 파이프라인이 어떻게 협력해서 binaural 입력을 처리하는지 보여줍니다.

세부 다이어그램은 다음 문서로 분리되어 있습니다.

- [01-2. TSE 파이프라인 상세](./tse-pipeline.md)
- [01-3. SED 파이프라인 상세](./sed-pipeline.md)

---

## 전체 데이터 흐름

```
                             ┌──────────────────────────────────────┐
                             │ BinauralCuratedDataset               │
                             │  scaper_fmt/{train,val,test}/{class}/│
                             │  bg_scaper_fmt/...                   │
                             │  noise_scaper_fmt/{train,val,test}/  │
                             │  hrtf/CIPIC/{*.sofa, *_hrtf.txt}     │
                             └─────────────────┬────────────────────┘
                                               │ on-the-fly
                                               ▼
                             ┌──────────────────────────────────────┐
                             │ SoundscapeDataset.__getitem__         │
                             │ (src/datasets/soundscape_dataset.py:421)│
                             │  - sample fg/bg/noise                │
                             │  - LUFS normalize                    │
                             │  - CIPIC HRTF convolution            │
                             └─────────────────┬────────────────────┘
                                               │
                                               ▼
                          ┌───────────────────────────────────────────┐
                          │ inputs = {                                │
                          │   "mixture":      (B, 2, 80000),          │
                          │   "label_vector": (B, 20),  ← multi-hot   │
                          │ }                                         │
                          │ targets = {                               │
                          │   "target":       (B, num_out, 80000),    │
                          │   "fg_labels":    list[str],              │
                          │ }                                         │
                          └────┬─────────────────────────┬────────────┘
                               │                         │
                               ▼                         ▼
                  ┌────────────────────┐    ┌──────────────────────────┐
                  │ SED                │    │ TSE                      │
                  │ ASTHuggingFace     │    │ Net  (TFGridNet Large)   │
                  │ src/sed/ast_hf.py  │    │ src/tse/net.py:31        │
                  │                    │    │   ├ STFT (chunk=96)      │
                  │ ┌────────────────┐ │    │   ├ MultiFiLMGuidedTFNet │
                  │ │ logits/scores  │ │    │   │   ├ Conv2d (proj)    │
                  │ │ softmax → 20cls│ │    │   │   ├ FiLM × 6 (preset │
                  │ └────────────────┘ │    │   │   │   = "all")       │
                  │         │          │    │   │   ├ GridNetBlock × 6 │
                  │         ▼          │    │   │   │   (LSTM intra +  │
                  │ ┌────────────────┐ │    │   │   │    LSTM inter)   │
                  │ │ multi-hot      │ │    │   │   └ ConvTranspose2d  │
                  │ │ label_vector   │─┼───→│   └ iSTFT + overlap-add  │
                  │ │ (B, 20)        │ │    │                          │
                  │ └────────────────┘ │    └────────────┬─────────────┘
                  └────────────────────┘                 │
                                                         ▼
                                       ┌────────────────────────────────┐
                                       │ output = {                     │
                                       │   "output":     (B, 5, 80000), │
                                       │   "next_state": dict,          │
                                       │ }                              │
                                       └────────────────────────────────┘
```

---

## 두 파이프라인의 역할 분담

| 파이프라인 | 입력 | 출력 | 역할 |
|---|---|---|---|
| **SED** | binaural mixture | 20-class multi-hot label vector | "지금 어떤 소리들이 들어있나?" 감지 |
| **TSE** | binaural mixture + label vector | per-source 분리된 1ch 파형 | "선택된 소리만 뽑아내라" 분리 |

> **데이터 흐름 핵심**: SED → label_vector → TSE 의 FiLM conditioning 으로 들어감.
> 실제로 학습/평가 시에는 ground-truth label_vector 를 사용하지만, 추론 시에는 SED 출력이 입력이 됨.

---

## 디렉토리 구조 (코드 기준)

```
fine_grained_soundscape_control_for_augmented_hearing/
├── configs/
│   ├── tse/                  ← TSE 학습 config (yaml)
│   │   ├── orange_pi.yaml    ← TFGridNet Large (본 문서 기준)
│   │   ├── raspberry_pi.yaml
│   │   ├── neuralaid.yaml
│   │   └── test_pipeline.yaml
│   └── sed/
│       └── ast_finetune.yaml
├── data/
│   ├── setup_data.py         ← 데이터셋 다운/구축 진입점
│   ├── pipeline/             ← Stage 1-3 (download/collect/prepare)
│   │   ├── download.py
│   │   ├── collect.py        ← per-source train/val/test.csv 생성
│   │   ├── prepare.py        ← scaper_fmt symlink + HRTF split
│   │   └── sources/          ← per-dataset 클래스 (FSD50K, ESC-50, ...)
│   ├── class_map.yaml        ← 20 클래스 정의 + AudioSet ID 매핑
│   └── ontology.json         ← AudioSet ontology
├── src/
│   ├── tse/
│   │   ├── net.py                       ← STFT wrapper (Net)
│   │   ├── multiflim_guided_tfnet.py    ← FiLM-conditioned separator
│   │   ├── gridnet_block.py             ← intra/inter LSTM 블록
│   │   ├── film.py                      ← FiLM(x, emb) = x*a(emb)+b(emb)
│   │   ├── train.py                     ← TSE 학습 진입점
│   │   ├── eval.py                      ← TSE 평가 진입점
│   │   └── model.py                     ← HF pretrained 로더
│   ├── sed/
│   │   ├── ast_hf.py                    ← HF AST 래퍼
│   │   ├── train.py                     ← SED 학습 진입점
│   │   └── eval.py                      ← SED 평가 + threshold 보정
│   ├── datasets/
│   │   ├── MisophoniaDataset.py         ← legacy on-the-fly 합성 (paper reproduction용)
│   │   ├── soundscape_dataset.py        ← clean reimpl, train.py 기본 (Dataset 직접 상속)
│   │   ├── multi_ch_simulator.py        ← CIPICSimulator (HRTF convolution)
│   │   └── motion_simulator.py
│   ├── trainer/
│   │   ├── base.py                      ← TrainerBackend ABC
│   │   ├── lightning.py                 ← PyTorch Lightning 백엔드
│   │   └── fabric.py                    ← Lightning Fabric 백엔드
│   └── metrics/
│       ├── tse.py                       ← SI-SDR, SNR
│       └── sed.py                       ← mAP, F1
└── docs/presentation/                   ← (이 문서 묶음)
```

---

## 한 줄 핵심 모델 요약 (TFGridNet Large)

| 항목 | 값 |
|---|---|
| 클래스 | `MultiFiLMGuidedTFNet` (`src/tse/multiflim_guided_tfnet.py:20`) |
| 블록 타입 | `GridNetBlock` (`src/tse/gridnet_block.py:6`) |
| 파라미터 | **501,738 (0.502 M)** |
| latent_dim D | 32 |
| hidden_channels H | 64 |
| num_layers B | 6 |
| FiLM preset | `"all"` (모든 6개 블록 앞에 FiLM 삽입) |
| 입력 채널 | 2 (binaural) |
| 출력 채널 | 5 |
| STFT chunk / lookback / lookahead | 96 / 96 / 64 samples (6/6/4 ms @ 16 kHz) |
| nfft | 256 (→ 129 freq bin) |

다음 단계: [TSE 파이프라인 상세 →](./tse-pipeline.md)
