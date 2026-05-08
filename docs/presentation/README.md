# Fine-Grained Soundscape Control — Handoff Documentation

This document set provides a comprehensive overview of the **TSE/SED pipeline** in this repository, designed so that a new maintainer can grasp the entire system at once. The reference model is the paper's **TFGridNet Large** (`configs/tse/orange_pi.yaml`, D=32, H=64, B=6, 5-output).

## At a glance

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
                                          │ Extracted target sources   │
                                          │ shape: (B, 5, 80000)       │
                                          └────────────────────────────┘
```

---

## Table of contents

| # | Section | Contents |
|---|---|---|
| 1 | [01-architecture/](./01-architecture/overview.md) | Overall system ASCII block diagram + internal structure of TSE/SED |
| 2 | [02-preprocessing/](./02-preprocessing/overview.md) | Offline split (CSV) -> online batch (binaural synthesis) |
| 3 | [03-training/](./03-training/overview.md) | Trainer backend, model layers, wandb usage examples |
| 4 | [04-samples/](./04-samples/sound-samples.ipynb) | Notebook with samples for the 20 classes plus TAU noise |

**Related external documents** (outside `presentation/`):

- [`docs/architecture.md`](../architecture.md) — Canonical architecture reference
- [`docs/pretrained_models.md`](../pretrained_models.md) — List of pretrained models on HF

---

## Entry points at a glance

| Task | Command | Code |
|---|---|---|
| Data preparation | `bash scripts/setup_dataset.sh --output_dir <path>` | `data/setup_data.py` |
| Training (TSE Large) | `python -m src.tse.train --config configs/tse/orange_pi.yaml --data_dir <path>` | `src/tse/train.py:207` (`main()`) |
| Training (SED) | `python -m src.sed.train --config configs/sed/ast_finetune.yaml --data_dir <path>` | `src/sed/train.py` |
| Evaluation (reproduce paper Table 3) | `python -m src.tse.eval --pretrained ooshyun/fine_grained_soundscape_control --model orange_pi --data_dir <path>` | `src/tse/eval.py` |
| Mini validation (5 minutes) | `bash scripts/eval/eval_mini.sh <data_dir>` | — |

---

## Key numbers (TFGridNet Large)

| Item | Value | Where measured |
|---|---|---|
| Parameters | 501,738 (0.502M) | `Net` (`src/tse/net.py:31`) |
| Forward MACs (5s, B=1) | 4.61 G | measured with fvcore |
| Forward MACs/s | 0.92 G | measured with fvcore |
| STFT chunk | 96 samples (6 ms) | `configs/tse/orange_pi.yaml:9` |
| nfft | 256 (= 129 freq bins) | `Net.__init__:63` |
| Input | (B, 2, 80000) | binaural 5s @ 16 kHz |
| Output | (B, 5, 80000) | 5-source separation |
| samples/epoch | 20,000 | `configs/tse/orange_pi.yaml:38` |
| max_epochs | 200 | `configs/tse/orange_pi.yaml:43` |
| batch_size | 8 | `configs/tse/orange_pi.yaml:35` |
| **Training VRAM (B=8 FP32)** | **~34 GB (extrapolated from measurements)** | — |
| **Training VRAM (B=8 AMP)** | ~18 GB | — |
| Recommended GPU | A100 40GB / L40S x 2 DDP | — |

---

## End-to-end data + execution flow

A unified view spanning dataset download, training, and evaluation. Every step references the corresponding code with `file_path:line_number`. The **streaming buffers (used for real-time inference)** and the **loss / metric branches** are also annotated.

### A. Offline stage — data classification -> preprocessing (run once)

```
$ bash scripts/setup_dataset.sh --output_dir <path>
                │
                ▼ (invoked internally)
        data/setup_data.py:11  main()
                │
                ▼
        data/pipeline/__init__.py:13  run(output_dir, stage='all', ...)
                │  random.seed(0); np.random.seed(0)   <- reproducibility
                │
                │  ┌─────────────────────────────────────────────────────────┐
                │  │ Stage 1: download                                       │
                │  │   data/pipeline/__init__.py:39                          │
                │  │   from .download import run_download                    │
                │  │   calls source.download(raw_dir) for each source        │
                │  │     ├ FSD50K  : data/pipeline/sources/fsd50k.py:23      │
                │  │     ├ ESC-50  : data/pipeline/sources/esc50.py          │
                │  │     ├ DISCO   : data/pipeline/sources/disco.py          │
                │  │     ├ musdb18 : data/pipeline/sources/musdb18.py        │
                │  │     ├ TAU     : data/pipeline/sources/tau.py            │
                │  │     └ CIPIC   : data/pipeline/sources/cipic.py          │
                │  │   -> raw/{dataset_name}/                                │
                │  └─────────────────────────────────────────────────────────┘
                │
                │  ┌─────────────────────────────────────────────────────────┐
                │  │ Stage 2: collect — data classification (CSV split)      │
                │  │   data/pipeline/collect.py:6  run_collect               │
                │  │     missing = []                                        │
                │  │     for source in sources:                              │
                │  │       1) source.try_use_reference_csvs()                │
                │  │            base.py:26 (prefer HF reference_splits)      │
                │  │       2) if raw is missing: missing.append(name)     :24│
                │  │          and continue                                   │
                │  │       3) source.collect(raw_dir, curated_dir)           │
                │  │     if missing:                                         │
                │  │       allow_missing=True  -> emit warning only          │
                │  │       allow_missing=False -> raise FileNotFoundError :37│
                │  │                                                         │
                │  │   For FSD50K:                                           │
                │  │     fsd50k.py:124  _collect_zenodo                      │
                │  │       quality filter: pp_pnp_ratings (>=2 pos, 0 neg)   │
                │  │       single-label only                                 │
                │  │       fsd50k.py:148  train_test_split(test_size=0.1,    │
                │  │                       stratify=label, random_state=42)  │
                │  │       common_labels intersection                        │
                │  │     base.py:52  _write_csvs() -> train/val/test.csv     │
                │  │                                                         │
                │  │   -> curated/{dataset}/{train,val,test}.csv             │
                │  │      columns: [fname, label, id]                        │
                │  └─────────────────────────────────────────────────────────┘
                │
                │  ┌─────────────────────────────────────────────────────────┐
                │  │ Stage 3: prepare — Scaper-format symlinks               │
                │  │   data/pipeline/prepare.py:252  run_prepare             │
                │  │                                                         │
                │  │   Step 1: build_id2classname()                          │
                │  │     prepare.py:30                                       │
                │  │     class_map.yaml + ontology subtree -> 20-class map   │
                │  │                                                         │
                │  │   Step 2: write_scaper_source() per (dataset, split)    │
                │  │     prepare.py:105                                      │
                │  │     for row in CSV:                                     │
                │  │       if row.id in id2classname:                        │
                │  │           target = scaper_fmt/{split}/{class}/          │
                │  │       elif is_valid_background(...):                    │
                │  │           target = bg_scaper_fmt/{split}/{label}/       │
                │  │       silence trim -> start_times.csv metadata          │
                │  │       os.symlink("../../../{dataset}/{fname}", target)  │
                │  │                                                         │
                │  │   Step 3: prepare_hrtf()                                │
                │  │     prepare.py:191                                      │
                │  │     CIPIC SOFA 80:10:10 split                           │
                │  │     -> hrtf/CIPIC/{train,val,test}_hrtf.txt             │
                │  │                                                         │
                │  │   Step 4: persist start_times.csv                       │
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

### B. Training stage — data load -> model -> loss / metric

```
$ python -m src.tse.train --config configs/tse/orange_pi.yaml --data_dir <path>
                │
                ▼
        src/tse/train.py:207  main()
                │  yaml.safe_load(args.config)
                │
                ├─ _build_datasets(cfg, data_dir)            src/tse/train.py:32
                │     SoundscapeDataset(split="train", ...)  <- inherits Dataset directly,
                │                                               on-the-fly binaural synthesis
                │
                ├─ DataLoader(train_ds, batch_size=8,        src/tse/train.py:241
                │             num_workers=8, pin_memory=True, shuffle=True)
                │
                ├─ _build_model(cfg) -> Net(...)             src/tse/train.py:89
                │
                ├─ optimizer/scheduler  (dotted-path)        src/tse/train.py:271-277
                │
                ├─ MultiResoFuseLoss(l1_ratio=10)            src/tse/loss.py:6
                │
                └─ trainer = create_trainer("lightning")     src/trainer/__init__.py:6
                  trainer.fit(model, train_loader, val_loader, loss_fn, opt, sched, cfg, metrics_fn,
                              resume_from=args.resume_from)   # <- new in Phase 2


┌─────────────────────────────────────────────────────────────────────────┐
│ 1) Data load (every step, in parallel across num_workers=8 processes)   │
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
│   │   │   ┌─ 🟡 Power filter (reject silent segments) ──────────────┐   │
│   │   │   │  _get_random_snippet(...)            soundscape_dataset.py:276│
│   │   │   │  pwr_threshold = self.pwr_threshold (= -40 dB)        :291│  │
│   │   │   │  while True:                                                │
│   │   │   │    audio = sample_snippet(...)                             │
│   │   │   │    pwr_dB = 10*log10(mean(audio²) + 1e-9)                  │
│   │   │   │    if not require_power: return                       :298│  │
│   │   │   │    if pwr_dB > pwr_threshold: return                  :301│  │
│   │   │   │    on retries exceeded, relax pwr_threshold by -1 dB :304│   │
│   │   │   └────────────────────────────────────────────────────────┘   │
│   │   │                                                                 │
│   │   │   sample_snippet(): sf.read + resample (torchaudio or librosa) │
│   │   └ label_vector[src_id] = 1  <- multi-hot                       :346│
│   │                                                                     │
│   ├ BG / noise via _get_random_snippet                          :356/369│
│   │   * noise: require_power=False (urban noise is low-energy)       :369│
│   │                                                                     │
│   ├ LUFS normalize                                              :381-389│
│   │   fg_lufs = ref_db + U[5,15]; bg_lufs = ref_db + U[0,10]            │
│   │   noise = ref_db                                                    │
│   │   _normalize_to_lufs(audio, sr, target_lufs)                  :62  │
│   │     <- iterative pyloudnorm correction                              │
│   │                                                                     │
│   ├ hrtf_simulator.simulate(fg+bg, noise, seed)                     :396│
│   │   CIPIC HRTF convolution -> binaural sources                        │
│   │                                                                     │
│   ├ mixture = sum(bi_srcs) + bi_noise                                :399│
│   │                                                                     │
│   └ gt[i] = mono(bi_srcs[i])  for i in range(n_fg)               :405-410│
│                                                                         │
│   ▼ (train path only)                                                   │
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
                ▼ (DataLoader assembles a batch)
   batch:
     inputs["mixture"]:      (8, 2, 80000)
     inputs["label_vector"]: (8, 20)        <- ground-truth label during training
     targets["target"]:      (8, 5, 80000)
                │
                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 2) Lightning training_step (one step over 8 samples)                    │
│   src/trainer/lightning.py:50                                           │
│                                                                         │
│   inputs, targets = batch                                               │
│   outputs = self.model(inputs)              <- calls Net.forward        │
│   loss = self.loss_fn(outputs, targets, inputs)   <- 3-arg adapter      │
│   self.log("train/loss", loss)                                          │
└─────────────────────────────────────────────────────────────────────────┘
                │
                ▼
```

---

### C. Inside the model — block-by-block detail + streaming buffers

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│ Net.forward(inputs, input_state, pad)              src/tse/net.py:216           │
│                                                                                 │
│   x   = inputs["mixture"]      # (B=8, M=2, t=80000)                            │
│   emb = inputs["embedding"]    # (B=8, 20)  <- label_vector                     │
│                                                                                 │
│   ┌── 🔵 Streaming buffer init (when input_state is None) ─────────┐            │
│   │   src/tse/net.py:222                                            │            │
│   │   input_state = self.init_buffers(B, device)        net.py:94   │            │
│   │     ├ tfnet_bufs = self.tfgridnet.init_buffers(B, device)       │            │
│   │     │     multiflim_guided_tfnet.py:154                         │            │
│   │     │     ├ conv_buf:    (B, 4, t_ksize-1=2, F=129)             │            │
│   │     │     ├ deconv_buf:  (B, 32, t_ksize-1=2, F=129)            │            │
│   │     │     └ block_bufs[i] = blocks[i].init_buffers(B, device)   │            │
│   │     │           gridnet_block.py:67                             │            │
│   │     │           ├ h0: (1, B*F, H=64)   <- inter-LSTM hidden     │            │
│   │     │           └ c0: (1, B*F, H=64)   <- inter-LSTM cell       │            │
│   │     └ istft_buf: (B*nO, synthesis_window_len, istft_lookback)   │            │
│   │           net.py:98                                              │            │
│   └─────────────────────────────────────────────────────────────────┘            │
│                                                                                 │
│   ▼                                                                             │
│   predict(x, embedding, input_state, pad=True)         net.py:186               │
│   │                                                                             │
│   ├ mod_pad(x, chunk_size=96, pad=(96, 64))            net.py:17                │
│   │   x: (B, 2, 80000) -> (B, 2, 80000+pad)                                     │
│   │                                                                             │
│   ├ extract_features(x)                                net.py:107               │
│   │   torch.stft(n_fft=256, hop=96, win=256)                                    │
│   │   -> (B, 4, T=836, F=129)   # 4 = 2 mic x (real, imag)                      │
│   │                                                                             │
│   ▼                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │ MultiFiLMGuidedTFNet.forward(batch, embedding, input_state)             │   │
│   │   src/tse/multiflim_guided_tfnet.py:169                                 │   │
│   │                                                                         │   │
│   │   # 🔵 use conv_buf — prepend the last (t_ksize-1) frames               │   │
│   │   #   carried over from the previous chunk                              │   │
│   │   batch = torch.cat((conv_buf, batch), dim=2)               :191        │   │
│   │   # 🔵 update conv_buf — keep the last (t_ksize-1) frames               │   │
│   │   #   of the current batch                                              │   │
│   │   conv_buf = batch[:, :, -(t_ksize-1):, :]                  :193        │   │
│   │                                                                         │   │
│   │   batch = self.conv(batch)            # (B, D=32, T, F)     :195        │   │
│   │     [Conv2d(4, 32, 3x3, padding=(0,1))]                                 │   │
│   │                                                                         │   │
│   │   # transform embedding (in this model embedding=None, so pass-through) │   │
│   │   if self.embedding is not None: embedding = self.embedding(embedding)  │   │
│   │                                                                  :209   │   │
│   │                                                                         │   │
│   │   for ii in range(self.n_layers):    # 6 blocks                  :215   │   │
│   │   ┌───────────────────────────────────────────────────────────┐         │   │
│   │   │ Block ii (FiLM + GridNetBlock)                            │         │   │
│   │   │                                                           │         │   │
│   │   │ # FiLM (preset="all" -> applied at every ii)      :218,224│         │   │
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
│   │   │ │   ── 1) intra-frame block (frequency axis) ──        │  │         │   │
│   │   │ │      gridnet_block.py:97-117                         │  │         │   │
│   │   │ │      reshape -> (B*T, Q, C)                          │  │         │   │
│   │   │ │      LayerNorm(C=32)                          :108   │  │         │   │
│   │   │ │      LSTM(in=32, hid=H=64, bidir=True)        :109   │  │         │   │
│   │   │ │        * stateless: hidden re-zeroed each call       │  │         │   │
│   │   │ │          (intra-frame)                               │  │         │   │
│   │   │ │      Linear(2H=128 -> C=32)                  :110    │  │         │   │
│   │   │ │      reshape -> (B, T, Q, C)                         │  │         │   │
│   │   │ │      + residual                              :118    │  │         │   │
│   │   │ │                                                      │  │         │   │
│   │   │ │   ── 2) inter-frame block (time axis, causal) ──     │  │         │   │
│   │   │ │      gridnet_block.py:121-153                        │  │         │   │
│   │   │ │      LayerNorm(C=32)                          :125   │  │         │   │
│   │   │ │                                                      │  │         │   │
│   │   │ │      # 🔵 use buffer (the heart of streaming!)       │  │         │   │
│   │   │ │      h0 = init_state["h0"]                    :127   │  │         │   │
│   │   │ │      c0 = init_state["c0"]                    :128   │  │         │   │
│   │   │ │                                                      │  │         │   │
│   │   │ │      transpose -> (B*Q, T, C)                 :132   │  │         │   │
│   │   │ │      inter_rnn, (h0, c0) = self.inter_rnn(           │  │         │   │
│   │   │ │          inter_rnn, (h0, c0))                 :136   │  │         │   │
│   │   │ │        LSTM(in=32, hid=64, bidir=False)              │  │         │   │
│   │   │ │                                                      │  │         │   │
│   │   │ │      # 🔵 update buffer — preserve for next chunk    │  │         │   │
│   │   │ │      init_state["h0"] = h0                    :146   │  │         │   │
│   │   │ │      init_state["c0"] = c0                    :147   │  │         │   │
│   │   │ │                                                      │  │         │   │
│   │   │ │      Linear(64 -> 32)                         :149   │  │         │   │
│   │   │ │      reshape -> (B, T, Q, C) + residual       :151   │  │         │   │
│   │   │ │                                                      │  │         │   │
│   │   │ │   return out, init_state                             │  │         │   │
│   │   │ └─────────────────────────────────────────────────────┘  │         │   │
│   │   └───────────────────────────────────────────────────────────┘         │   │
│   │   ...Block 0...1...2...3...4...5  (6 iterations)                        │   │
│   │                                                                         │   │
│   │   permute -> (B, D=32, T, F)                                 :236       │   │
│   │                                                                         │   │
│   │   # 🔵 use / update deconv_buf (same pattern as conv_buf)               │   │
│   │   batch = torch.cat((deconv_buf, batch), dim=2)              :238       │   │
│   │   deconv_buf = batch[:, :, -(t_ksize-1):, :]                 :240       │   │
│   │                                                                         │   │
│   │   batch = self.deconv(batch)         # (B, S*2=10, T, F)     :242       │   │
│   │     [ConvTranspose2d(32, 10, 3x3)]                                      │   │
│   │   reshape -> (B, S=5, T, 2*F=258)     split real/imag        :243       │   │
│   │                                                                         │   │
│   │   # 🔵 store all buffers in input_state and return                      │   │
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
│   │   reshape + irfft  -> (B*S, iW, T)             net.py:148-150               │
│   │   apply synthesis_window                       net.py:154                   │
│   │                                                                             │
│   │   # 🔵 use istft_buf — prepend the residual from the previous chunk         │
│   │   #   for overlap-add                                                       │
│   │   x = torch.cat([istft_buf, x], dim=-1)        net.py:159                   │
│   │   # 🔵 update istft_buf                                                     │
│   │   istft_buf = x[..., -istft_buf.shape[1]:]     net.py:160                   │
│   │                                                                             │
│   │   F.fold (overlap-add)                         net.py:163                   │
│   │   drop pad samples + reshape                   net.py:174-180               │
│   │                                                                             │
│   │   return x: (B, S=5, t=80000), input_state                                  │
│                                                                                 │
│   if pad: x = x[..., :-mod] (trim mod_pad)                  net.py:211          │
│                                                                                 │
│   return {"output": x, "next_state": input_state}          net.py:227           │
└─────────────────────────────────────────────────────────────────────────────────┘
```

> 🔵 **Summary of the 5 streaming buffers**
> - `conv_buf` (prepend before input conv): `multiflim_guided_tfnet.py:191/193/248`
> - `deconv_buf` (prepend before output deconv): `multiflim_guided_tfnet.py:238/240/249`
> - `block_bufs[i].h0, c0` (inter-LSTM hidden/cell): `gridnet_block.py:127/128/146/147`
> - `istft_buf` (overlap-add): `net.py:159/160`
>
> During training, `input_state=None` is passed in and the state is reinitialized on every forward (no chunked processing). **At inference (streaming) time, maintaining the state externally** enables true real-time processing.

---

### D. Loss vs. metric — branch

```
                outputs = model(inputs)
                est = outputs["output"]   # (B, 5, 80000)
                gt  = targets["target"]   # (B, 5, 80000)
                          │
            ┌─────────────┴──────────────┐
            ▼                            ▼
┌────────────────────────────┐  ┌───────────────────────────────────┐
│ Training path              │  │ Validation / evaluation path      │
│ (training_step)            │  │ For scoring (no back-prop)        │
│ Used for Lightning back-   │  │ src/trainer/lightning.py:60       │
│ prop                       │  │                                   │
│ src/trainer/lightning.py:50│  │ if metrics_fn is not None:        │
│                            │  │   metrics = metrics_fn(           │
│ loss = self.loss_fn(       │  │       outputs, targets, inputs)   │
│     outputs, targets, inputs)  │ <- TSE adapter:                  │
│ <- TSE adapter wraps       │  │   est = outputs["output"]         │
│   MultiResoFuseLoss(       │  │   gt  = targets["target"]         │
│     est=outputs["output"], │  │   mix = inputs["mixture"]         │
│     gt=targets["target"])  │  │   num_fg = targets["num_fg_labels"]│
│  tse/train.py:_build_loss_fn│ │   mask = arange(C) < num_fg       │
│   = MR-STFT + 10*L1        │  │   -> si_sdri / snri (active only) │
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
                            │  -> save best-{epoch}.ckpt            │
                            │                                       │
                            │ ReduceLROnPlateau                     │
                            │  scheduler.step(val/loss)             │
                            │                                       │
                            │ (optional) EarlyStopping              │
                            │  patience=20                          │
                            └──────────────────────────────────────┘
```

---

### E. Standalone evaluation (after training) — separate evaluation flow

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
                │     # per-label loop when nO==1, single forward when nO>1
                │     # (TFGridNet Large has nO=5 -> single forward)
                │
                ▼
        compute_metrics_tse(gt, est, mix, label_vector,    src/tse/eval.py:234
                            metric_func_list=[snr_i,
                                              snr_per_channel,
                                              si_sdr, si_sdr_i,
                                              si_sdr_per_channel, ...])
                │
                ▼
        Output artifacts:
          {output_dir}/results.csv               (per-sample)
          {output_dir}/metrics_total_averages.json  (averages)
```

---

### Stage -> code mapping at a glance

| Stage | Entry point | Key code |
|---|---|---|
| Data classification (CSV split) | `bash scripts/setup_dataset.sh` | `data/pipeline/collect.py:6` + `sources/fsd50k.py:148` (90:10 stratified) |
| Preprocessing (scaper-format) | same as above | `data/pipeline/prepare.py:252` (4-step) |
| Data load (binaural synthesis) | `DataLoader` (worker x 8) | `src/datasets/soundscape_dataset.py:421` (`__getitem__`) |
| Model forward entry | `Net.forward` | `src/tse/net.py:216` |
| Model buffer init | `Net.init_buffers` | `src/tse/net.py:94` |
| Conv entry | `MultiFiLMGuidedTFNet.forward` | `src/tse/multiflim_guided_tfnet.py:195` |
| FiLM application | same as above | `src/tse/multiflim_guided_tfnet.py:218,224` + `film.py:11` |
| GridNetBlock intra-frame | `GridNetBlock.forward` | `src/tse/gridnet_block.py:97-117` |
| GridNetBlock inter-frame (causal) | same as above | `src/tse/gridnet_block.py:121-153` |
| Streaming buffer (h0, c0) | same as above | `src/tse/gridnet_block.py:127/128/146/147` |
| Deconv output | `MultiFiLMGuidedTFNet.forward` | `src/tse/multiflim_guided_tfnet.py:242` |
| iSTFT (overlap-add) | `Net.synthesis` | `src/tse/net.py:136` |
| Loss computation | `_LitWrapper.training_step` | `src/trainer/lightning.py:53` + `src/tse/train.py:_build_loss_fn` |
| Metric computation | `_LitWrapper.validation_step` | `src/trainer/lightning.py:67` + `src/tse/train.py:_build_metrics_fn` |
| Standalone evaluation | `python -m src.tse.eval` | `src/tse/eval.py:145, 234` |
