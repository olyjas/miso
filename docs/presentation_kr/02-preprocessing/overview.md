# 02-1. 전처리 — 개요

본 레포의 전처리는 **두 단계** 입니다.

| 단계 | 시점 | 결과물 | 상세 |
|---|---|---|---|
| **A. 오프라인 split** | 학습 시작 전 1회 | `BinauralCuratedDataset/scaper_fmt/{train,val,test}/{class}/` 심볼릭 링크 + 각 데이터셋의 `{train,val,test}.csv` | [02-2 →](./dataset-split.md) |
| **B. 온라인 batch** | 매 epoch 마다 sample 단위 | `(mixture, target, label_vector)` 텐서 | [02-3 →](./batch-creation.md) |

---

## A 단계 vs B 단계 — 무엇이 어디서 일어나는가

```
┌──────────────────────────────────────────────────────────────────────────┐
│ A. 오프라인 (1회 실행)                                                   │
│   bash scripts/setup_dataset.sh --output_dir <path>                      │
│   ├─ Stage 1: download   FSD50K, ESC-50, musdb18, DISCO, TAU, CIPIC       │
│   ├─ Stage 2: collect    각 데이터셋 → train/val/test.csv                │
│   └─ Stage 3: prepare    scaper_fmt/{class}/ 심링크, HRTF split,         │
│                          start_times.csv (silence trim 메타)             │
│                                                                          │
│   결과:                                                                  │
│   <path>/BinauralCuratedDataset/                                         │
│     ├ scaper_fmt/{train,val,test}/{20-classes}/         ← FG 음원        │
│     ├ bg_scaper_fmt/{train,val,test}/{class}/           ← BG 음원        │
│     ├ noise_scaper_fmt/{train,val,test}/{scene}/        ← TAU urban noise│
│     ├ hrtf/CIPIC/{train,val,test}_hrtf.txt + *.sofa     ← HRTF split     │
│     └ start_times.csv                                                    │
└──────────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ B. 온라인 (학습 중 매 sample)                                            │
│   SoundscapeDataset.__getitem__(idx)                                     │
│   src/datasets/soundscape_dataset.py:421                                 │
│                                                                          │
│   1. RNG seed = idx + np.random.randint(1e6)  (train 시)                 │
│   2. n_fg = U[1,5], n_bg = U[1,3], n_noise = 1                           │
│   3. for each fg/bg:                                                     │
│        - random class → random file → random 5s snippet                  │
│        - 🟡 power 필터: pwr_dB > -40 dB 인 snippet 만 채택               │
│             (_get_random_snippet, line 301)                              │
│             → 이벤트 없는 무음 구간 reject (재시도 초과 시 -1 dB 완화)  │
│        - resample to 16kHz                                               │
│        - LUFS normalize (target = -50 + SNR_fg/bg)                       │
│      for noise:                                                          │
│        - require_power=False (도시 소음은 그대로, line 369)              │
│   4. hrtf_simulator.simulate(sources, noise, seed)                       │
│        → binaural rendering with HRTF convolution                        │
│   5. mixture = sum(bi_srcs) + bi_noise                                   │
│   6. peak normalize                                                      │
│   7. AudioAugmentations.apply_random_augmentations  (train 시만)         │
│                                                                          │
│   returns:                                                               │
│   inputs  = {"mixture": (2, 80000), "label_vector": (20,)}               │
│   targets = {"target": (num_out, 80000), "fg_labels": [...], ...}        │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 데이터셋 라이센스 및 출처

| Dataset | 용도 | 라이선스 | 위치 |
|---|---|---|---|
| FSD50K | FG (다수 클래스) | Mixed CC | `raw/FSD50K/` |
| ESC-50 | FG/BG | CC-BY-NC 3.0 | `raw/ESC-50/` |
| musdb18 | FG (`singing`, `music`) | Academic only | `raw/musdb18/` |
| DISCO | BG noise | CC-BY 4.0 | `raw/disco_noises/` |
| TAU-2019 | Noise (urban scene) | NC | `raw/TAU-acoustic-sounds/` |
| CIPIC HRTF | binaural simulator | Public Domain | `raw/CIPIC-HRTF/` 또는 `cipic-hrtf-database/` |

라이선스 제약 때문에 **단일 tar 로 재배포 불가** → 본 레포는 자동 다운로드 + 사전 빌드된 메타데이터(`data/prebuilt/metadata.tar.gz`)를 함께 제공합니다.

---

## 시간 추정 (참고용)

| 단계 | 디스크 | 시간 (네트워크 100 Mbps 기준) |
|---|---|---|
| Stage 1 download (전체) | ~125 GB | 2-4 시간 |
| Stage 2 collect | < 1 MB CSV | 수 분 |
| Stage 3 prepare (심링크 + silence trim) | symlinks only | 30-60 분 (NFS 위에선 수 시간) |
| 미니 데이터셋 (검증용) | ~250 MB | 5 분 (`scripts/eval/eval_mini.sh`) |

다음 단계: [A. 오프라인 split 상세 →](./dataset-split.md)
