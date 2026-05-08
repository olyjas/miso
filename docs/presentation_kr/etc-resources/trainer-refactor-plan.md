# Trainer Refactor Plan — train + val + 사전학습 로드

관련 문서: [`docs/architecture.md`](./architecture.md) · [`docs/pretrained_models.md`](./pretrained_models.md)

본 문서는 **`src/trainer/`** 의 현재 결함을 정리하고, 두 단계 (시그니처 통일 → 사전학습 모델 로드) 의 개발 계획을 기록합니다.

> **상태 (2026-05-08)**: ✅ **Phase 1 + Phase 2 + Phase 3 모두 적용 완료** (commits `7a3ad68`, `b9fb250`, `e6e7560`).
> 본 문서의 "결함" / "변경 범위" 절은 **착수 전 상태** 를 보존한 기록이고, 현재 코드는 모든 매트릭스가 동작합니다 (실측 결과는 §7 참조).

> **범위**: train + val 만 다룸. test 는 `src/{tse,sed}/eval.py` 에서 별도 처리.

---

## 0. 현재 상태 진단 — 정리

| 조합 | 동작 | 원인 |
|---|---|---|
| **Lightning + TSE** | ✅ | `_LitWrapper` 가 `outputs["output"]`/`targets["target"]` 추출 + `loss_fn(est=, gt=)` 호출, MultiResoFuseLoss 가 이 시그니처 수용 |
| **Lightning + SED** | ❌ | 같은 추출 + `loss_fn(est=, gt=)` 호출 → MultiLabelBCELoss(logits, targets) 시그니처 불일치 |
| **Fabric + TSE** | ❌ | `loss_fn(outputs_dict, targets_dict)` 로 dict 통째 전달 → MultiResoFuseLoss 가 텐서 기대 → AttributeError |
| **Fabric + SED** | ❌ | 동일 dict 전달 → BCE 가 텐서 기대 → 동일 |

### 핵심 결함 3가지

- **결함 A** — Fabric 의 `loss_fn(outputs, targets)` (dict 통째) ↔ Lightning 의 `loss_fn(est=, gt=)` (텐서 추출 후 kwargs) 불일치 (`fabric.py:42, 91, 98` ↔ `lightning.py:55, 67`)
- **결함 B** — Lightning step 이 TSE 키 (`"output"`, `"target"`) + 시그니처 (`est=, gt=`) 에 강결합. SED 못 씀.
- **결함 C** — `metrics_fn` arity 가 backend 별로 다름. Lightning `(est, gt, mix)` 3-arg vs Fabric `(outputs, targets)` 2-arg. TSE `_metrics_fn` 은 3-arg, SED 는 2-arg → 매트릭스 전체가 어긋남.

---

## 1. Phase 1 — Trainer 시그니처 통일

**목표**: 양쪽 backend 가 동일한 callable 인터페이스를 받음. dict→tensor 추출은 **train.py 의 adapter** 가 책임.

### 1-A. Trainer 호출 규약 (양쪽 통일)

```python
# 양쪽 backend 의 train/val step:
outputs = model(inputs)
loss    = loss_fn(outputs, targets)              # (dict, dict) → tensor
metrics = metrics_fn(outputs, targets, inputs)   # (dict, dict, dict) → dict[str, scalar]
```

→ trainer 는 **키를 모름**. 추출은 adapter 가 함.

### 1-B. Train.py 가 adapter 만들어 넘김

**TSE**:
```python
def _build_loss_fn(cfg):
    base = MultiResoFuseLoss(l1_ratio=cfg["loss"].get("l1_ratio", 10))
    def adapter(outputs, targets):
        return base(est=outputs["output"], gt=targets["target"]).mean()
    return adapter

def _build_metrics_fn():
    def adapter(outputs, targets, inputs):
        return _metrics_fn(outputs["output"], targets["target"], inputs["mixture"])
    return adapter
```

**SED**:
```python
def _build_loss_fn(cfg, dataset, device):
    base = get_loss_function(cfg, dataset=dataset, device=device)
    def adapter(outputs, targets):
        # SED 의 GT 는 inputs["labels"] (multi-hot) — targets 가 아님 주의
        return base(outputs["output"], targets["target"]).mean()
        #            ↑ logits          ↑ 또는 inputs["labels"]
    return adapter
```

> ⚠️ **결정 필요 항목** (1-B 안에서):
> SED 의 GT 가 `targets["target"]` (waveform) 인지 `inputs["labels"]` (multi-hot) 인지 결정. BCE 입장에서는 **multi-hot 이 맞음**. trainer 가 inputs 도 같이 넘기는 게 자연스러움.

### 1-C. 변경 파일

| 파일 | 변경 |
|---|---|
| `src/trainer/base.py` | `fit/validate` 시그니처 명시: `loss_fn(outputs, targets) → tensor`, `metrics_fn(outputs, targets, inputs) → dict` |
| `src/trainer/lightning.py:50-74` | `_LitWrapper.training_step/validation_step` 의 dict 추출 로직 제거. `loss_fn(outputs, targets)` 직호출 |
| `src/trainer/fabric.py:20-122` | `loss_fn(outputs, targets)` 그대로 유지, `metrics_fn` 호출 시 inputs 추가 |
| `src/tse/train.py:121, 196, 201` | `_metrics_fn` → `_build_metrics_fn`, loss → adapter wrapping |
| `src/sed/train.py:113, 206, 213` | 동일 패턴, GT 키는 `inputs["labels"]` 로 결정 시 그쪽 사용 |

### 1-D. Phase 1 결과 매트릭스

| 조합 | 변경 후 |
|---|---|
| Lightning + TSE | ✅ (회귀 없음) |
| Lightning + SED | ✅ |
| Fabric + TSE | ✅ |
| Fabric + SED | ✅ |

---

## 2. Phase 2 — 사전학습 모델 로드 / 학습 재개

**목표**: 두 가지 사용 사례 지원.

### 2-A. 사용 사례

| Use Case | 의미 | CLI 인자 |
|---|---|---|
| **Init from pretrained** | 가중치만 로드 (optimizer/scheduler 새로 생성, epoch 0 부터) — fine-tuning | `--init_from <spec>` |
| **Resume training** | 모델 + optimizer + scheduler + epoch 모두 복원 — crash recovery | `--resume_from <path>` |

### 2-B. `--init_from` 입력 형식 3가지 (Phase 3 검증 결과 기반)

| 형식 | 처리 | 예시 |
|---|---|---|
| `"<repo_id>:<model_name>"` | `load_pretrained()` 호출 → **HF config 로 자체 빌드** + 가중치 로드. 사용자 yaml `model:` 섹션 무시 + 경고 로그 | `ooshyun/fine_grained_soundscape_control:orange_pi_film_all` |
| `"path/to/foo.ckpt"` (Lightning) | `torch.load` → `ckpt["state_dict"]` → `"model."` 접두사 제거 → yaml 빌드 모델에 strict load | `runs/tse/best-epoch=42.ckpt` |
| `"path/to/foo.pt"` (raw) | `torch.load` → `state_dict`/`model_state_dict`/본체 자동 탐지 → yaml 빌드 모델에 load | `runs/tse/best.pt` |

### 2-C. `--resume_from` 처리

| Backend | 처리 |
|---|---|
| Lightning | `pl.Trainer.fit(lit_model, ..., ckpt_path=resume_from)` — Lightning native 가 모든 state 자동 복원 |
| Fabric | `torch.load` → 수동으로 model/optimizer state 복원 + `start_epoch = ckpt["epoch"] + 1` 로 루프 시작 |

### 2-D. CLI 인터페이스 (TSE/SED 양쪽 동일)

```python
parser.add_argument(
    "--init_from", type=str, default=None,
    help='Initialize weights only. Formats:\n'
         '  "<repo_id>:<model_name>"  → HF (e.g. ooshyun/foo:orange_pi_film_all)\n'
         '  "path/to/foo.ckpt"        → Lightning ckpt\n'
         '  "path/to/foo.pt"          → raw or wrapped state_dict')
parser.add_argument(
    "--resume_from", type=str, default=None,
    help="Resume full training (model + optim + scheduler + epoch).")
```

`--init_from` 과 `--resume_from` 은 **mutually exclusive**.

### 2-E. main() 흐름

```
parse args
  │
  ▼
if args.init_from:
    if "<repo>:<name>" 패턴 and not local path:
        model = load_pretrained(repo, name)        # HF config 로 자체 빌드
        logger.warning("yaml 'model' section ignored — using HF config")
    else:
        model = _build_model(cfg)                  # yaml 로 빌드
        sd = torch.load(args.init_from, weights_only=False)
        sd = _normalize_state_dict(sd)             # Lightning 접두사 처리 등
        model.load_state_dict(sd, strict=True)
else:
    model = _build_model(cfg)
  │
  ▼
optimizer = _instantiate(...)                     # 항상 새로 생성
scheduler = _instantiate(...)                     # 항상 새로 생성
loss_fn   = _build_loss_fn(cfg)                   # adapter
metrics_fn = _build_metrics_fn()                  # adapter
  │
  ▼
trainer = create_trainer(backend)
trainer.fit(
    model, train_loader, val_loader,
    loss_fn, optimizer, scheduler, cfg, metrics_fn,
    resume_from=args.resume_from,                  # ← Phase 2 신규
)
```

### 2-F. Backend `fit` 시그니처 변경

```python
# src/trainer/base.py
@abstractmethod
def fit(
    self,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_fn: Callable,
    optimizer: Optimizer,
    scheduler: _LRScheduler,
    config: dict,
    metrics_fn: Callable | None = None,
    resume_from: str | None = None,    # ← Phase 2 신규
) -> dict[str, Any]: ...
```

---

## 3. Phase 3 — HF 호환성 검증 결과

`use_first_ln` 차이 발견 + 수정 완료 (이 문서와 같이 commit).

### 3-A. 발견 — yaml ↔ HF 모델 구조 불일치

| 항목 | configs/tse/orange_pi.yaml (수정 전) | HF `orange_pi_film_all` |
|---|---|---|
| `use_first_ln` | **false** | **true** |
| 결과 | strict load 실패 (`ln.weight/bias` 키 누락) | — |

### 3-B. 수정

`use_first_ln: false → true` 일괄 적용:
- `configs/tse/orange_pi.yaml`
- `configs/tse/raspberry_pi.yaml`
- `configs/tse/neuralaid.yaml`
- `configs/tse/test_pipeline.yaml`

### 3-C. 검증 결과 (수정 후)

```
params: pretrained=501,802, fresh=501,802, match=True ✓
strict=True load OK ✓
forward output max abs diff = 0.00e+00 (bit-exact) ✓
```

→ HF orange_pi_film_all 의 가중치를 yaml 기반 fresh Net 에 로드해 **forward 결과 완전 일치** 확인.

### 3-D. HF 모델 ↔ yaml 매핑 (참고)

| yaml (이 레포) | 매치되는 HF entry | 비고 |
|---|---|---|
| `orange_pi.yaml` (5ch, 5out, film_all) | `orange_pi_film_all` | Phase 3 검증 ✓ |
| `raspberry_pi.yaml` (현재 5out 셋업) | (대응 entry 없음 — `raspberry_pi` 는 1ch/1out) | 5out 학습 시 fresh start |
| `neuralaid.yaml` (현재 GridNetBlock) | (대응 entry 없음 — `neuralaid` 는 MLPBlock) | **별도 결함**, Phase 4 후보 |
| `test_pipeline.yaml` | — | 검증 전용 (HF 로드 필요 X) |

> ⚠️ raspberry_pi.yaml / neuralaid.yaml 의 모델 구조 자체가 HF entry 와 일치하지 않을 수 있음. **Phase 2 의 `--init_from` 사용 시 5ch/5out 으로 학습된 HF entry 가 없는 모델은 fresh init 부터** 진행.

---

## 4. 작업 체크리스트 (착수 순서)

### Phase 1 — Trainer 시그니처 통일 ✅ (commit `7a3ad68`)
- [x] `src/trainer/base.py` 의 `fit/validate` 시그니처 docstring 명시
- [x] `src/trainer/lightning.py` `_LitWrapper.training_step` / `validation_step` 단순화 (dict 추출 제거)
- [x] `src/trainer/fabric.py` `_train_one_epoch` / `_validate_one_epoch` 의 `metrics_fn` 호출에 `inputs` 추가
- [x] `src/tse/train.py` 에 `_build_loss_fn`, `_build_metrics_fn` adapter 추가, `main()` 에서 사용
- [x] `src/sed/train.py` 동일 패턴 + SED GT 키 결정 — `inputs["labels"]` 사용 (D1)
- [x] 4 매트릭스 (L+T, L+S, F+T, F+S) 1 epoch 학습 smoke test (§7-A)

### Phase 2 — 사전학습 로드 ✅ (commit `b9fb250`)
- [x] `src/trainer/base.py` `fit(..., resume_from=None)` 인자 추가
- [x] `src/trainer/lightning.py` `fit()` 가 `pl.Trainer.fit(ckpt_path=resume_from)` 위임
- [x] `src/trainer/fabric.py` `fit()` 가 manual resume (model + optimizer state + start_epoch)
- [x] `src/tse/train.py` `--init_from`, `--resume_from` argparse 추가
- [x] `normalize_state_dict()` helper — `src/trainer/__init__.py` (Lightning 접두사 / wrapped dict / raw state 자동 처리)
- [x] HF spec 파싱 (`<repo>:<model_name>`) 로직
- [x] `src/sed/train.py` 동일
- [x] HF init smoke test (§7-B)
- [x] Lightning ckpt resume smoke test (§7-C)

### Phase 3 — yaml ↔ HF 호환 ✅ (commit `3409127`)
- [x] HF orange_pi_film_all 다운로드 + load 검증
- [x] `use_first_ln` 4 yaml 일괄 수정 (false → true)

---

## 5. 미결 결정 항목

| # | 결정 항목 | 후보 | 권장 |
|---|---|---|---|
| D1 | SED 의 BCE GT 출처 | `targets["target"]` (waveform) vs `inputs["labels"]` (multi-hot) | **inputs["labels"]** (BCE 의미상 정답) |
| D2 | `--init_from <hf>` 시 yaml model 처리 | (a) HF config 로 자체 빌드 + warning / (b) yaml strict 검증 + 실패 시 에러 | **(a)**, 사용자에게 명시적 warning |
| D3 | metrics_fn 시그니처 | (a) 3-arg `(out, tgt, inp)` 통일 / (b) 2-arg `(out, tgt)` + adapter 가 inputs 캡처 | **(a)**, trainer 가 inputs 도 알기 쉬움 |
| D4 | raspberry_pi/neuralaid yaml 의 5out 셋업 유지? | 현재 셋업 유지 vs 1out 으로 되돌려 HF entry 매칭 | **현재 유지**, fresh 학습 의도일 수 있음 |

---

## 6. 변경 영향 범위

| 영역 | 영향 |
|---|---|
| 학습 스크립트 | tse/sed train.py 진입점 인자 변경 (backward-compatible 유지 가능) |
| Backend | trainer/{base,lightning,fabric}.py 시그니처 + 내부 로직 변경 |
| Configs | yaml 의 model 섹션은 그대로. `use_first_ln: true` 일괄 변경 (Phase 3) |
| 평가 | `src/{tse,sed}/eval.py` 영향 없음 (별도 흐름) |
| 외부 호환 | HF 사전학습 모델은 모두 그대로 로드 가능 (검증 ✓) |

---

## 7. 실측 검증 결과 (2026-05-08)

mini dataset 으로 1 epoch × 4 samples 매트릭스 + HF init + resume 모두 PASS.

### 7-A. 4 backend×task 매트릭스

| 시나리오 | train_loss | val_loss | 비고 |
|---|---|---|---|
| Lightning + TSE | 4.65 | 8.30 | si_sdri/snri 정상 |
| Fabric + TSE | 5.28 | 8.02 | 동일 |
| Lightning + SED | 0.97 | 0.65 | mAP 등 정상 |
| Fabric + SED | 0.18 | 0.41 | 동일 |

### 7-B. `--init_from` HF (`orange_pi_film_all`) → 1 epoch fine-tune

```
val/si_sdri = 42.9 dB
val/snri    = 43.4 dB
```
사전학습 가중치가 그대로 활용되어 random init 대비 큰 차이 확인.

### 7-C. `--resume_from` Lightning ckpt

```
INFO: Restored all states from the checkpoint at runs/test/tse_lightning/best-epoch=00.ckpt
Epoch 1/1 — train/loss=9.65 val/loss=9.80
```
Lightning native 의 `ckpt_path=` 를 통한 model + optimizer + scheduler + epoch 모두 복원됨 확인.

### 7-D. 검증 도중 발견된 사전 결함 (모두 fix 됨, commit `e6e7560`)

| 결함 | 위치 | 수정 |
|---|---|---|
| TSE `embedding` 키 = int idx (FiLM 은 multi-hot float 기대) | `soundscape_dataset.py` | `embedding = label_vector` |
| `fg_labels` 가변 길이 collate 실패 | `soundscape_dataset.py` | `num_fg_max` "None" padding |
| Fabric `max_epochs` 가 cfg["training"] 안 읽음 | `fabric.py` | Lightning 식 fallback |
| SED `data_dir` 강제 prefix 가 yaml 무시 | `sed/train.py` | TSE 식 relative re-root |
| SED `_build_dataset` split/HRTF 미매핑 | `sed/train.py` | `{split}` 자동 join |
| ASTModel 잘못된 kwargs (`num_classes/freeze_encoder/sample_rate`) | `sed/train.py` | `num_labels` + `freeze_model()` |
| `get_trainable_parameters()` 미구현 메서드 | `sed/train.py` | `sum(p.numel())` |
| `load_pretrained()` 후 eval 모드 → cudnn LSTM backward 거부 | `tse/train.py` | `model.train()` 추가 |
