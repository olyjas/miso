# Fine-Grained Soundscape Control — 인계 문서

이 문서 묶음은 본 레포의 **TSE/SED 파이프라인 전체 구조**를 신규 인계자가 한 번에 파악할 수 있도록 정리한 자료입니다. 모델 기준은 논문의 **TFGridNet Large** (`configs/tse/orange_pi.yaml`, D=32, H=64, B=6, 5-output) 입니다.

## 한 장 요약

```
                        ┌─────────────────────────────────────────────┐
                        │           Binaural Mixture (2ch, 16kHz)     │
                        │           shape: (B, 2, 80000)              │
                        └────────────────┬────────────────────────────┘
                                         │
                       ┌─────────────────┴─────────────────┐
                       ▼                                   ▼
            ┌──────────────────────┐         ┌─────────────────────────┐
            │   SED Pipeline       │         │   TSE Pipeline          │
            │   ASTHuggingFace     │         │   Net (TFGridNet Large) │
            │   src/sed/ast_hf.py  │         │   src/tse/net.py        │
            └──────────┬───────────┘         └────────────┬────────────┘
                       │                                  ▲
                       │ label_vector (multi-hot, 20-dim) │
                       └──────────────────────────────────┘
                                                          │
                                                          ▼
                                          ┌────────────────────────────┐
                                          │ 분리된 타깃 음원            │
                                          │ shape: (B, 5, 80000)        │
                                          └────────────────────────────┘
```

---

## 목차

| # | 섹션 | 내용 |
|---|---|---|
| 1 | [01-architecture/](./01-architecture/overview.md) | 전체 시스템 ASCII 블록 다이어그램 + TSE/SED 내부 구조 |
| 2 | [02-preprocessing/](./02-preprocessing/overview.md) | 오프라인 split (CSV) → 온라인 batch (binaural 합성) |
| 3 | [03-training/](./03-training/overview.md) | Trainer 백엔드, 모델 레이어, wandb 사용 예제 |
| 4 | [04-samples/](./04-samples/sound-samples.ipynb) | 20 클래스 + TAU 노이즈 샘플 노트북 |

**관련 외부 문서** (presentation 외):

- [`docs/architecture.md`](../architecture.md) — 정식 아키텍처 레퍼런스
- [`docs/pretrained_models.md`](../pretrained_models.md) — HF 사전학습 모델 목록

---

## 진입점 한눈에 보기

| 작업 | 명령 | 코드 |
|---|---|---|
| 데이터 준비 | `bash scripts/setup_dataset.sh --output_dir <path>` | `data/setup_data.py` |
| 학습 (TSE Large) | `python -m src.tse.train --config configs/tse/orange_pi.yaml --data_dir <path>` | `src/tse/train.py:207` (`main()`) |
| 학습 (SED) | `python -m src.sed.train --config configs/sed/ast_finetune.yaml --data_dir <path>` | `src/sed/train.py` |
| 평가 (논문 Table 3 재현) | `python -m src.tse.eval --pretrained ooshyun/fine_grained_soundscape_control --model orange_pi --data_dir <path>` | `src/tse/eval.py` |
| 미니 검증 (5분) | `bash scripts/eval/eval_mini.sh <data_dir>` | — |

---

## 핵심 수치 (TFGridNet Large 기준)

| 항목 | 값 | 측정 위치 |
|---|---|---|
| 파라미터 | 501,738 (0.502M) | `Net` (`src/tse/net.py:31`) |
| Forward MACs (5s, B=1) | 4.61 G | fvcore 측정 |
| Forward MACs/s | 0.92 G | fvcore 측정 |
| STFT chunk | 96 samples (6 ms) | `configs/tse/orange_pi.yaml:9` |
| nfft | 256 (=129 freq bin) | `Net.__init__:63` |
| 입력 | (B, 2, 80000) | binaural 5s @ 16 kHz |
| 출력 | (B, 5, 80000) | 5-source separation |
| samples/epoch | 20,000 | `configs/tse/orange_pi.yaml:38` |
| max_epochs | 200 | `configs/tse/orange_pi.yaml:43` |
| batch_size | 8 | `configs/tse/orange_pi.yaml:35` |
| **학습 VRAM (B=8 FP32)** | **~34 GB (실측 외삽)** | — |
| **학습 VRAM (B=8 AMP)** | ~18 GB | — |
| 권장 GPU | A100 40GB / L40S × 2 DDP | — |

---

## End-to-End 데이터 + 실행 흐름

데이터셋 다운로드부터 학습/평가까지 한 번에 보는 통합 흐름입니다. 각 단계는 `파일경로:라인` 으로 코드를 직접 짚어줍니다. **streaming buffer (real-time 처리용)** 와 **loss / metric 분기** 도 함께 표시.

### A. 오프라인 단계 — 데이터 분류 → 전처리 (1회 실행)

```
$ bash scripts/setup_dataset.sh --output_dir <path>
                │
                ▼ (내부적으로 호출)
        data/setup_data.py:11  main()
                │
                ▼
        data/pipeline/__init__.py:13  run(output_dir, stage='all', ...)
                │  random.seed(0); np.random.seed(0)   ← 재현성
                │
                │  ┌─────────────────────────────────────────────────────────┐
                │  │ Stage 1: download                                       │
                │  │   data/pipeline/__init__.py:39                          │
                │  │   from .download import run_download                    │
                │  │   각 source.download(raw_dir) 호출                      │
                │  │     ├ FSD50K  : data/pipeline/sources/fsd50k.py:23      │
                │  │     ├ ESC-50  : data/pipeline/sources/esc50.py          │
                │  │     ├ DISCO   : data/pipeline/sources/disco.py          │
                │  │     ├ musdb18 : data/pipeline/sources/musdb18.py        │
                │  │     ├ TAU     : data/pipeline/sources/tau.py            │
                │  │     └ CIPIC   : data/pipeline/sources/cipic.py          │
                │  │   → raw/{dataset_name}/                                 │
                │  └─────────────────────────────────────────────────────────┘
                │
                │  ┌─────────────────────────────────────────────────────────┐
                │  │ Stage 2: collect — 데이터 분류 (CSV split)              │
                │  │   data/pipeline/collect.py:6  run_collect               │
                │  │     missing = []                                        │
                │  │     for source in sources:                              │
                │  │       1) source.try_use_reference_csvs()                │
                │  │            base.py:26 (HF reference_splits 우선)         │
                │  │       2) raw 없으면 missing.append(name) + continue   :24│
                │  │       3) source.collect(raw_dir, curated_dir)           │
                │  │     if missing:                                         │
                │  │       allow_missing=True  → warning 만 출력              │
                │  │       allow_missing=False → raise FileNotFoundError  :37│
                │  │                                                         │
                │  │   FSD50K 의 경우:                                       │
                │  │     fsd50k.py:124  _collect_zenodo                      │
                │  │       quality filter: pp_pnp_ratings (≥2 pos, 0 neg)    │
                │  │       single-label only                                 │
                │  │       fsd50k.py:148  train_test_split(test_size=0.1,    │
                │  │                       stratify=label, random_state=42)  │
                │  │       common_labels intersection                        │
                │  │     base.py:52  _write_csvs() → train/val/test.csv      │
                │  │                                                         │
                │  │   → curated/{dataset}/{train,val,test}.csv              │
                │  │      columns: [fname, label, id]                        │
                │  └─────────────────────────────────────────────────────────┘
                │
                │  ┌─────────────────────────────────────────────────────────┐
                │  │ Stage 3: prepare — Scaper-format 심링크                 │
                │  │   data/pipeline/prepare.py:252  run_prepare             │
                │  │                                                         │
                │  │   Step 1: build_id2classname()                          │
                │  │     prepare.py:30                                       │
                │  │     class_map.yaml + ontology subtree → 20-class map    │
                │  │                                                         │
                │  │   Step 2: write_scaper_source() per (dataset, split)    │
                │  │     prepare.py:105                                      │
                │  │     for row in CSV:                                     │
                │  │       if row.id in id2classname:                        │
                │  │           target = scaper_fmt/{split}/{class}/          │
                │  │       elif is_valid_background(...):                    │
                │  │           target = bg_scaper_fmt/{split}/{label}/       │
                │  │       silence trim → start_times.csv 메타               │
                │  │       os.symlink("../../../{dataset}/{fname}", target)  │
                │  │                                                         │
                │  │   Step 3: prepare_hrtf()                                │
                │  │     prepare.py:191                                      │
                │  │     CIPIC SOFA 80:10:10 split                           │
                │  │     → hrtf/CIPIC/{train,val,test}_hrtf.txt              │
                │  │                                                         │
                │  │   Step 4: start_times.csv 저장                          │
                │  │     prepare.py:319                                      │
                │  └─────────────────────────────────────────────────────────┘
                │
                ▼
   <output>/BinauralCuratedDataset/
     ├ scaper_fmt/{train,val,test}/{20-class}/*.wav        (symlink)
     ├ bg_scaper_fmt/{train,val,test}/{class}/
     ├ noise_scaper_fmt/{train,val,test}/{scene}/          (TAU symlink)
     ├ hrtf/CIPIC/{*.sofa, train_hrtf.txt, ...}
     └ start_times.csv
```

---

### B. 학습 단계 — 데이터 로드 → 모델 → loss / metric

```
$ python -m src.tse.train --config configs/tse/orange_pi.yaml --data_dir <path>
                │
                ▼
        src/tse/train.py:207  main()
                │  yaml.safe_load(args.config)
                │
                ├─ _build_datasets(cfg, data_dir)            src/tse/train.py:32
                │     SoundscapeDataset(split="train", ...)  ← Dataset 직접 상속, on-the-fly binaural 합성
                │
                ├─ DataLoader(train_ds, batch_size=8,        src/tse/train.py:241
                │             num_workers=8, pin_memory=True, shuffle=True)
                │
                ├─ _build_model(cfg) → Net(...)              src/tse/train.py:89
                │
                ├─ optimizer/scheduler  (dotted-path)        src/tse/train.py:271-277
                │
                ├─ MultiResoFuseLoss(l1_ratio=10)            src/tse/loss.py:6
                │
                └─ trainer = create_trainer("lightning")     src/trainer/__init__.py:6
                  trainer.fit(model, train_loader, val_loader, loss_fn, opt, sched, cfg, metrics_fn,
                              resume_from=args.resume_from)   # ← Phase 2 신규


┌─────────────────────────────────────────────────────────────────────────┐
│ 1) 데이터 로드 (매 step, num_workers=8 worker 프로세스에서 병렬)         │
│                                                                         │
│   SoundscapeDataset.__getitem__(idx)   src/datasets/soundscape_dataset.py:421 │
│   │                                                                     │
│   │ seed = idx + np.random.randint(1e6)  if split=="train"          :423│
│   │ rng = np.random.RandomState(seed)                                   │
│   │                                                                     │
│   ▼                                                                     │
│   create_scene(rng)                                                  :313│
│   │                                                                     │
│   ├ n_fg = rng.randint(num_fg_range)                                 :320│
│   │   n_bg = rng.randint(num_bg_range); n_noise = rng.randint(...)      │
│   │                                                                     │
│   ├ for _ in range(n_fg):                                            :334│
│   │   ├ src_id = rng.randint(0, len(fg_sounds))                         │
│   │   ├ audio = _get_random_snippet(fg_dir, label,                      │
│   │   │                              rng, require_power=True)        :339│
│   │   │                                                                 │
│   │   │   ┌─ 🟡 Power 필터 (이벤트 없는 구간 reject) ───────────────┐   │
│   │   │   │  _get_random_snippet(...)            soundscape_dataset.py:276│
│   │   │   │  pwr_threshold = self.pwr_threshold (= -40 dB)       :291│   │
│   │   │   │  while True:                                                │
│   │   │   │    audio = sample_snippet(...)                              │
│   │   │   │    pwr_dB = 10*log10(mean(audio²) + 1e-9)                   │
│   │   │   │    if not require_power: return                       :298│  │
│   │   │   │    if pwr_dB > pwr_threshold: return                  :301│  │
│   │   │   │    재시도 초과 시 pwr_threshold -= 1 dB 로 완화        :304│  │
│   │   │   └────────────────────────────────────────────────────────┘   │
│   │   │                                                                 │
│   │   │   sample_snippet() : sf.read + resample (torchaudio 또는 librosa)│
│   │   └ label_vector[src_id] = 1  ← multi-hot                       :346│
│   │                                                                     │
│   ├ BG / noise 는 _get_random_snippet 으로                       :356/369│
│   │   ※ noise: require_power=False (도시 소음 원래 작음)             :369│
│   │                                                                     │
│   ├ LUFS normalize                                              :381-389│
│   │   fg_lufs = ref_db + U[5,15]; bg_lufs = ref_db + U[0,10]            │
│   │   noise = ref_db                                                    │
│   │   _normalize_to_lufs(audio, sr, target_lufs)                  :62  │
│   │     ← pyloudnorm 반복 보정                                          │
│   │                                                                     │
│   ├ hrtf_simulator.simulate(fg+bg, noise, seed)                     :396│
│   │   CIPIC HRTF convolution → binaural sources                         │
│   │                                                                     │
│   ├ mixture = sum(bi_srcs) + bi_noise                                :399│
│   │                                                                     │
│   └ gt[i] = mono(bi_srcs[i])  for i in range(n_fg)               :405-410│
│                                                                         │
│   ▼ (train 경로만)                                                      │
│   perturbations.apply_random_augmentations(mixture, target, rng)    :433│
│                                                                         │
│   peak normalize: if mixture.abs().max() > 1: divide              :438-441│
│                                                                         │
│   return inputs={"mixture": (2, 80000), "label_vector": (20,),     :448│
│                  "embedding": (20,)},                                    │
│          targets={"target": (5, 80000), "fg_labels": [...],         :463│
│                   "num_fg_labels": int}                                  │
└─────────────────────────────────────────────────────────────────────────┘
                │
                ▼ (DataLoader 가 batch 묶음)
   batch:
     inputs["mixture"]:      (8, 2, 80000)
     inputs["label_vector"]: (8, 20)        ← 학습 시 ground-truth label
     targets["target"]:      (8, 5, 80000)
                │
                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 2) Lightning training_step (8 sample 한 step)                           │
│   src/trainer/lightning.py:50                                           │
│                                                                         │
│   inputs, targets = batch                                               │
│   outputs = self.model(inputs)              ← Net.forward 호출          │
│   loss = self.loss_fn(outputs, targets, inputs)   ← 3-arg adapter        │
│   self.log("train/loss", loss)                                          │
└─────────────────────────────────────────────────────────────────────────┘
                │
                ▼
```

---

### C. 모델 내부 흐름 — 블록별 상세 + streaming buffer

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│ Net.forward(inputs, input_state, pad)              src/tse/net.py:216           │
│                                                                                 │
│   x   = inputs["mixture"]      # (B=8, M=2, t=80000)                            │
│   emb = inputs["embedding"]    # (B=8, 20)  ← label_vector                      │
│                                                                                 │
│   ┌── 🔵 Streaming Buffer 초기화 (input_state is None 일 때) ──────┐            │
│   │   src/tse/net.py:222                                            │            │
│   │   input_state = self.init_buffers(B, device)        net.py:94   │            │
│   │     ├ tfnet_bufs = self.tfgridnet.init_buffers(B, device)       │            │
│   │     │     multiflim_guided_tfnet.py:154                         │            │
│   │     │     ├ conv_buf:    (B, 4, t_ksize-1=2, F=129)             │            │
│   │     │     ├ deconv_buf:  (B, 32, t_ksize-1=2, F=129)            │            │
│   │     │     └ block_bufs[i] = blocks[i].init_buffers(B, device)   │            │
│   │     │           gridnet_block.py:67                             │            │
│   │     │           ├ h0: (1, B*F, H=64)   ← inter-LSTM hidden     │            │
│   │     │           └ c0: (1, B*F, H=64)   ← inter-LSTM cell       │            │
│   │     └ istft_buf: (B*nO, synthesis_window_len, istft_lookback)   │            │
│   │           net.py:98                                              │            │
│   └─────────────────────────────────────────────────────────────────┘            │
│                                                                                 │
│   ▼                                                                             │
│   predict(x, embedding, input_state, pad=True)         net.py:186               │
│   │                                                                             │
│   ├ mod_pad(x, chunk_size=96, pad=(96, 64))            net.py:17                │
│   │   x: (B, 2, 80000) → (B, 2, 80000+pad)                                      │
│   │                                                                             │
│   ├ extract_features(x)                                net.py:107               │
│   │   torch.stft(n_fft=256, hop=96, win=256)                                    │
│   │   → (B, 4, T=836, F=129)   # 4 = 2 mic × (real, imag)                       │
│   │                                                                             │
│   ▼                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │ MultiFiLMGuidedTFNet.forward(batch, embedding, input_state)             │   │
│   │   src/tse/multiflim_guided_tfnet.py:169                                 │   │
│   │                                                                         │   │
│   │   # 🔵 conv_buf 사용 — 직전 chunk 의 마지막 (t_ksize-1) 프레임 prepend │   │
│   │   batch = torch.cat((conv_buf, batch), dim=2)               :191        │   │
│   │   # 🔵 conv_buf 업데이트 — 현재 batch 끝 (t_ksize-1) 프레임 보존        │   │
│   │   conv_buf = batch[:, :, -(t_ksize-1):, :]                  :193        │   │
│   │                                                                         │   │
│   │   batch = self.conv(batch)            # (B, D=32, T, F)     :195        │   │
│   │     [Conv2d(4, 32, 3×3, padding=(0,1))]                                 │   │
│   │                                                                         │   │
│   │   # embedding 변환 (본 모델은 embedding=None 이라 그대로 통과)          │   │
│   │   if self.embedding is not None: embedding = self.embedding(embedding)  │   │
│   │                                                                  :209   │   │
│   │                                                                         │   │
│   │   for ii in range(self.n_layers):    # 6 블록                  :215    │   │
│   │   ┌───────────────────────────────────────────────────────────┐         │   │
│   │   │ Block ii (FiLM + GridNetBlock)                            │         │   │
│   │   │                                                           │         │   │
│   │   │ # FiLM (preset="all" → 모든 ii 에서 적용)         :218,224│         │   │
│   │   │ batch = self.film_layers[f"film_layer_{ii}"](batch, emb)  │         │   │
│   │   │   FiLM:  src/tse/film.py:11                                │         │   │
│   │   │     a = Linear_a(emb);  b = Linear_b(emb)                 │         │   │
│   │   │     return x * a + b                                      │         │   │
│   │   │                                                           │         │   │
│   │   │ # GridNetBlock                                            │         │   │
│   │   │ batch, gridnet_buf[f"buf{ii}"] = self.blocks[ii](          │         │   │
│   │   │     batch, gridnet_buf[f"buf{ii}"])              :232     │         │   │
│   │   │                                                           │         │   │
│   │   │ ┌─ GridNetBlock.forward(x, init_state)                ─┐  │         │   │
│   │   │ │   src/tse/gridnet_block.py:82                        │  │         │   │
│   │   │ │   x: (B, T, Q=129, C=32)                             │  │         │   │
│   │   │ │                                                      │  │         │   │
│   │   │ │   ── 1) intra-frame block (frequency 축) ──          │  │         │   │
│   │   │ │      gridnet_block.py:97-117                         │  │         │   │
│   │   │ │      reshape → (B*T, Q, C)                           │  │         │   │
│   │   │ │      LayerNorm(C=32)                          :108   │  │         │   │
│   │   │ │      LSTM(in=32, hid=H=64, bidir=True)        :109   │  │         │   │
│   │   │ │        ※ stateless: hidden 매번 0 초기화 (intra)    │  │         │   │
│   │   │ │      Linear(2H=128 → C=32)                    :110   │  │         │   │
│   │   │ │      reshape → (B, T, Q, C)                          │  │         │   │
│   │   │ │      + residual                              :118    │  │         │   │
│   │   │ │                                                      │  │         │   │
│   │   │ │   ── 2) inter-frame block (time 축, causal) ──       │  │         │   │
│   │   │ │      gridnet_block.py:121-153                        │  │         │   │
│   │   │ │      LayerNorm(C=32)                          :125   │  │         │   │
│   │   │ │                                                      │  │         │   │
│   │   │ │      # 🔵 buffer 사용 (streaming 핵심!)              │  │         │   │
│   │   │ │      h0 = init_state["h0"]                    :127   │  │         │   │
│   │   │ │      c0 = init_state["c0"]                    :128   │  │         │   │
│   │   │ │                                                      │  │         │   │
│   │   │ │      transpose → (B*Q, T, C)                  :132   │  │         │   │
│   │   │ │      inter_rnn, (h0, c0) = self.inter_rnn(           │  │         │   │
│   │   │ │          inter_rnn, (h0, c0))                 :136   │  │         │   │
│   │   │ │        LSTM(in=32, hid=64, bidir=False)              │  │         │   │
│   │   │ │                                                      │  │         │   │
│   │   │ │      # 🔵 buffer 업데이트 — 다음 chunk 위해 보존     │  │         │   │
│   │   │ │      init_state["h0"] = h0                    :146   │  │         │   │
│   │   │ │      init_state["c0"] = c0                    :147   │  │         │   │
│   │   │ │                                                      │  │         │   │
│   │   │ │      Linear(64 → 32)                          :149   │  │         │   │
│   │   │ │      reshape → (B, T, Q, C) + residual        :151   │  │         │   │
│   │   │ │                                                      │  │         │   │
│   │   │ │   return out, init_state                             │  │         │   │
│   │   │ └─────────────────────────────────────────────────────┘  │         │   │
│   │   └───────────────────────────────────────────────────────────┘         │   │
│   │   …Block 0…1…2…3…4…5  (6 회 반복)                                       │   │
│   │                                                                         │   │
│   │   permute → (B, D=32, T, F)                                  :236       │   │
│   │                                                                         │   │
│   │   # 🔵 deconv_buf 사용 / 업데이트 (conv_buf 와 동일 패턴)               │   │
│   │   batch = torch.cat((deconv_buf, batch), dim=2)              :238       │   │
│   │   deconv_buf = batch[:, :, -(t_ksize-1):, :]                 :240       │   │
│   │                                                                         │   │
│   │   batch = self.deconv(batch)         # (B, S*2=10, T, F)     :242       │   │
│   │     [ConvTranspose2d(32, 10, 3×3)]                                      │   │
│   │   reshape → (B, S=5, T, 2*F=258)     real/imag 분리          :243       │   │
│   │                                                                         │   │
│   │   # 🔵 모든 buffer 를 input_state 에 저장 후 반환                       │   │
│   │   input_state["conv_buf"]   = conv_buf                       :248       │   │
│   │   input_state["deconv_buf"] = deconv_buf                     :249       │   │
│   │   input_state["block_bufs"] = gridnet_buf                    :250       │   │
│   │   return batch, input_state                                  :252       │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│   ▼                                                                             │
│   synthesis(batch, input_state)                            net.py:136           │
│   │                                                                             │
│   │   x: (B, S=5, T, 2F=258)                                                    │
│   │   reshape + irfft  → (B*S, iW, T)              net.py:148-150               │
│   │   apply synthesis_window                       net.py:154                   │
│   │                                                                             │
│   │   # 🔵 istft_buf 사용 — overlap-add 직전 chunk 잔차 prepend                │
│   │   x = torch.cat([istft_buf, x], dim=-1)        net.py:159                   │
│   │   # 🔵 istft_buf 업데이트                                                   │
│   │   istft_buf = x[..., -istft_buf.shape[1]:]     net.py:160                   │
│   │                                                                             │
│   │   F.fold (overlap-add)                         net.py:163                   │
│   │   drop pad samples + reshape                   net.py:174-180               │
│   │                                                                             │
│   │   return x: (B, S=5, t=80000), input_state                                  │
│                                                                                 │
│   if pad: x = x[..., :-mod] (mod_pad 만큼 트림)            net.py:211           │
│                                                                                 │
│   return {"output": x, "next_state": input_state}          net.py:227           │
└─────────────────────────────────────────────────────────────────────────────────┘
```

> 🔵 **streaming buffer 5종 요약**
> - `conv_buf` (입력 conv prepend): `multiflim_guided_tfnet.py:191/193/248`
> - `deconv_buf` (출력 deconv prepend): `multiflim_guided_tfnet.py:238/240/249`
> - `block_bufs[i].h0, c0` (inter-LSTM hidden/cell): `gridnet_block.py:127/128/146/147`
> - `istft_buf` (overlap-add): `net.py:159/160`
>
> 학습 시에는 `input_state=None` 으로 들어가 매 forward 마다 새로 초기화 (chunk 처리 아님). **추론(streaming) 시에 외부에서 state 를 유지** 하면 진짜 real-time 처리됨.

---

### D. Loss 계산 vs Metric 계산 — 분기

```
                outputs = model(inputs)
                est = outputs["output"]   # (B, 5, 80000)
                gt  = targets["target"]   # (B, 5, 80000)
                          │
            ┌─────────────┴──────────────┐
            ▼                            ▼
┌────────────────────────────┐  ┌───────────────────────────────────┐
│ 학습 경로 (training_step)  │  │ 검증/평가 경로                    │
│ Lightning back-prop 용     │  │ scoring 용 (back-prop X)          │
│ src/trainer/lightning.py:50│  │ src/trainer/lightning.py:60       │
│                            │  │                                   │
│ loss = self.loss_fn(       │  │ if metrics_fn is not None:        │
│     outputs, targets, inputs)  │   metrics = metrics_fn(           │
│ ← TSE adapter wraps        │  │       outputs, targets, inputs)   │
│   MultiResoFuseLoss(       │  │ ← TSE adapter:                    │
│     est=outputs["output"], │  │   est = outputs["output"]         │
│     gt=targets["target"])  │  │   gt  = targets["target"]         │
│  tse/train.py:_build_loss_fn│  │   mix = inputs["mixture"]         │
│   = MR-STFT + 10*L1        │  │   num_fg = targets["num_fg_labels"]│
│                            │  │   mask = arange(C) < num_fg       │
│                            │  │   → si_sdri / snri (active only)  │
│                            │  │  tse/train.py:_build_metrics_fn   │
│ loss.backward()            │  │                                   │
│ optimizer.step()           │  │ self.log("val/loss", ...)         │
│                            │  │ self.log("val/si_sdri", ...)      │
│ self.log("train/loss",     │  │ self.log("val/snri", ...)         │
│   loss, sync_dist=True)    │  │                                   │
└────────────────────────────┘  └───────────────────────────────────┘
                                            │
                                            ▼
                            ┌──────────────────────────────────────┐
                            │ ModelCheckpoint                       │
                            │  monitor="val/loss", mode="min"       │
                            │  src/trainer/lightning.py:115         │
                            │  → save best-{epoch}.ckpt             │
                            │                                       │
                            │ ReduceLROnPlateau                     │
                            │  scheduler.step(val/loss)             │
                            │                                       │
                            │ (선택) EarlyStopping                  │
                            │  patience=20                          │
                            └──────────────────────────────────────┘
```

---

### E. 평가 단독 실행 (학습 후) — 별도 evaluation flow

```
$ python -m src.tse.eval --pretrained ooshyun/fine_grained_soundscape_control \
                          --model orange_pi --data_dir <path>
                │
                ▼
        src/tse/eval.py:evaluate()                        src/tse/eval.py:145
                │
                │  for batch in test_loader:
                │     inputs["mixture"] (1, 2, 80000)
                │     inputs["embedding"] = label_vector (1, 20)
                │
                │     with torch.no_grad():
                │         outputs = model(inputs)           src/tse/eval.py:220
                │
                │     # nO==1 일 때 per-label loop, nO>1 일 때 single forward
                │     # (TFGridNet Large 는 nO=5 → single forward)
                │
                ▼
        compute_metrics_tse(gt, est, mix, label_vector,    src/tse/eval.py:234
                            metric_func_list=[snr_i,
                                              snr_per_channel,
                                              si_sdr, si_sdr_i,
                                              si_sdr_per_channel, ...])
                │
                ▼
        결과 저장:
          {output_dir}/results.csv               (per-sample)
          {output_dir}/metrics_total_averages.json  (평균)
```

---

### 한눈에 보는 단계 → 코드 매핑

| 단계 | 진입점 | 핵심 코드 |
|---|---|---|
| 데이터 분류 (CSV split) | `bash scripts/setup_dataset.sh` | `data/pipeline/collect.py:6` + `sources/fsd50k.py:148` (90:10 stratified) |
| 전처리 (scaper-format) | 동일 | `data/pipeline/prepare.py:252` (4-step) |
| 데이터 로드 (binaural 합성) | `DataLoader` (worker × 8) | `src/datasets/soundscape_dataset.py:421` (`__getitem__`) |
| 모델 forward 진입 | `Net.forward` | `src/tse/net.py:216` |
| 모델 buffer 초기화 | `Net.init_buffers` | `src/tse/net.py:94` |
| Conv 진입 | `MultiFiLMGuidedTFNet.forward` | `src/tse/multiflim_guided_tfnet.py:195` |
| FiLM 적용 | 동일 | `src/tse/multiflim_guided_tfnet.py:218,224` + `film.py:11` |
| GridNetBlock intra-frame | `GridNetBlock.forward` | `src/tse/gridnet_block.py:97-117` |
| GridNetBlock inter-frame (causal) | 동일 | `src/tse/gridnet_block.py:121-153` |
| streaming buffer (h0, c0) | 동일 | `src/tse/gridnet_block.py:127/128/146/147` |
| Deconv 출력 | `MultiFiLMGuidedTFNet.forward` | `src/tse/multiflim_guided_tfnet.py:242` |
| iSTFT (overlap-add) | `Net.synthesis` | `src/tse/net.py:136` |
| Loss 계산 | `_LitWrapper.training_step` | `src/trainer/lightning.py:53` + `src/tse/train.py:_build_loss_fn` |
| Metric 계산 | `_LitWrapper.validation_step` | `src/trainer/lightning.py:67` + `src/tse/train.py:_build_metrics_fn` |
| 평가 (별도) | `python -m src.tse.eval` | `src/tse/eval.py:145, 234` |
