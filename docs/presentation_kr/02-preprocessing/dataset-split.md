# 02-2. 오프라인 split — 데이터셋 → train/val/test.csv

[← 전처리 개요](./overview.md)

이 단계는 **여러 공개 데이터셋을 받아와 클래스별 train/val/test split CSV 와 scaper-format 심볼릭 링크 디렉토리** 를 만드는 일회성 작업입니다.

---

## 진입점

```bash
# 권장: 한 줄 실행
bash scripts/setup_dataset.sh --output_dir /path/to/data

# 내부적으로 호출되는 Python 진입점
python data/setup_data.py --output_dir <path> --stage all \
                          [--datasets fsd50k,esc50,disco,cipic,musdb18,tau] \
                          [--manual_dir <path>] \
                          [--reference_dir <path>]
```

전체 흐름 (`data/pipeline/__init__.py:13`):

```
run(output_dir, stage='all', datasets=None, ...)
  │ random.seed(0); np.random.seed(0)   ← 재현성을 위한 시드
  │
  ├─→ Stage 1: download
  │     data/pipeline/download.py:run_download
  │     각 source 의 BaseSource.download() 호출
  │
  ├─→ Stage 2: collect
  │     data/pipeline/collect.py:run_collect
  │     각 source 의 BaseSource.collect() 호출 → train/val/test.csv 생성
  │
  └─→ Stage 3: prepare
        data/pipeline/prepare.py:run_prepare
        CSV 읽어서 scaper_fmt/ symlink + HRTF split + start_times.csv
```

---

## Stage 1: download

`data/pipeline/download.py:run_download` 가 각 source 의 `download(raw_dir)` 메서드 호출.

| 데이터셋 | 다운로드 방식 | 위치 |
|---|---|---|
| FSD50K | HF mirror `Fhrozen/FSD50k` 자동 다운로드 (또는 Zenodo 수동) | `raw/FSD50K/` |
| ESC-50 | GitHub zip 자동 다운로드 | `raw/ESC-50/` |
| DISCO | Zenodo (4019030) 자동 | `raw/disco_noises/` |
| CIPIC | HF dataset `ooshyun/...` 또는 UC Davis | `raw/CIPIC-HRTF/` |
| **musdb18** | 수동 다운로드 필요 (license) | `raw/musdb18/` |
| **TAU-2019** | 수동 다운로드 (Zenodo 3063822, 11 audio + meta) | `raw/TAU-acoustic-sounds/` |

수동 다운로드가 필요한 데이터셋은 `--manual_dir <path>` 로 위치 지정.

---

## Stage 2: collect — per-source CSV 생성

`data/pipeline/collect.py:6` 의 `run_collect` 가 호출하는 `BaseSource` 의 핵심 인터페이스 (`data/pipeline/sources/base.py:8`):

```python
class BaseSource(ABC):
    name: str  # e.g. "FSD50K"
    key:  str  # e.g. "fsd50k"

    @abstractmethod
    def download(self, raw_dir: Path) -> None: ...

    @abstractmethod
    def collect(self, raw_dir: Path, curated_dir: Path) -> None:
        """Produce {train,val,test}.csv in curated_dir/{self.name}/."""
```

### 2-1. Reference CSV 우선 사용

`base.py:26` `try_use_reference_csvs()` — `--reference_dir` 가 주어지면 사전 검증된 CSV 들을 그대로 복사합니다 (HF `ooshyun/fine_grained_soundscape_control/reference_splits/` 에서 받음). 동일한 split 을 보장하기 위한 권장 경로입니다.

```python
def try_use_reference_csvs(self, curated_dir, reference_dir):
    if reference_dir is None: return False
    ref = reference_dir / (self.ref_csv_dir or self.name)
    csvs = [ref / f"{s}.csv" for s in ("train", "val", "test")]
    if not all(c.exists() for c in csvs): return False
    out = curated_dir / self.name; out.mkdir(parents=True, exist_ok=True)
    for c in csvs: shutil.copy2(c, out / c.name)
    return True
```

### 2-2. FSD50K 의 split 로직 (예시)

`data/pipeline/sources/fsd50k.py:124` (Zenodo 레이아웃 기준):

```
1. pp_pnp_ratings_FSD50K.json 로드 → quality filter
   → 각 라벨이 ≥2 positive, 0 negative/uncertain 인 샘플만 통과

2. collection_dev.csv → dev_curated (single-label only)
   collection_eval.csv → test_samples

3. dev_curated 를 라벨별 90:10 stratified split:
   sklearn.model_selection.train_test_split(test_size=0.1, random_state=42)
   → train_samples, val_samples

4. 세 split 모두에 등장하는 클래스만 유지 (common_labels intersection)

5. CSV 저장: train.csv / val.csv / test.csv
   columns: [fname, label, id]
```

다른 source 들의 split 규칙은 `data/pipeline/sources/{esc50,disco,musdb18,tau,cipic}.py` 참조.

### 2-3. CSV 출력 형식

`base.py:52` `_write_csvs()`:

```python
out_dir = curated_dir / self.name  # 예: curated/FSD50K/
for split, df in [("train", train), ("val", val), ("test", test)]:
    df.to_csv(out_dir / f"{split}.csv", index=False)
```

각 CSV 의 컬럼은 source 마다 다르지만 공통 키는:

| 컬럼 | 의미 |
|---|---|
| `fname` | dataset 내 상대 경로 (예: `FSD50K.dev_audio/12345.wav`) |
| `label` | AudioSet 클래스 이름 (예: `Dog`, `Bark`) |
| `id` | AudioSet ID (예: `/m/0bt9lr`) |

---

## Stage 3: prepare — Scaper-format symlink

`data/pipeline/prepare.py:252` `run_prepare` 가 4 단계로 진행:

### 3-1. id2classname 매핑 구축

`prepare.py:30` `build_id2classname()`:

```python
class_map: dict[str, list[str]] = yaml.safe_load(open("data/class_map.yaml"))
# 예: {"dog": ["Dog"], "cat": ["Cat", "Domestic cat"], ...}

id2classname = {}
for class_name, elements in class_map.items():
    for element_name in elements:
        node_id = ontology.get_id_from_name(element_name)
        # subtree 까지 모두 같은 class_name 으로 매핑
        for cid in ontology.get_subtree(node_id):
            id2classname[cid] = class_name
```

→ 결과: AudioSet ID → 20 class 중 하나로 매핑 (subtree 까지 포함).

### 3-2. Scaper-format symlink 생성

`prepare.py:105` `write_scaper_source()`:

```
for dataset in ["FSD50K", "ESC-50", "musdb18", "disco_noises"]:
    for split in ["train", "val", "test"]:
        df = pd.read_csv(curated/{dataset}/{split}.csv)
        for row in df:
            sample_id = row["id"]   # AudioSet ID
            if sample_id in id2classname:
                # foreground
                out_path = scaper_fmt/{split}/{class_name}/
            elif is_valid_background(sample_id, ...):
                # background (Music/Human voice 제외, FG 와 겹치지 않는)
                out_path = bg_scaper_fmt/{split}/{label}/
            else:
                continue

            # 상대 심링크 생성
            os.symlink("../../../{dataset_name}/{fname}", out_path/fname)

            # silence trim 메타
            start, first_silence, end = trim_silence(audio)
            all_samples.append({fname, start_sample, end_sample, first_silence})
```

### 3-3. HRTF split

`prepare.py:191` `prepare_hrtf()`:

```
1. CIPIC SOFA 파일들을 raw/CIPIC-HRTF/ 또는 raw/cipic-hrtf-database/ 에서 찾음
2. <output_dir>/hrtf/CIPIC/ 으로 복사
3. random.shuffle(copied) — random.seed(0) 로 재현성 보장
4. 80:10:10 split → train_hrtf.txt / val_hrtf.txt / test_hrtf.txt
```

### 3-4. start_times.csv 저장

silence trimming 메타데이터를 통합 저장:

```
output_dir/start_times.csv
  columns: [fname, start_sample, end_sample, first_silence]
```

학습 시 `SoundscapeDataset.sample_snippet()` 가 이 정보를 활용해 무음 구간을 피합니다.

---

## 최종 디렉토리 구조

```
/path/to/data/
├ raw/                                 ← Stage 1 결과 (raw downloads)
│   ├ FSD50K/, ESC-50/, musdb18/
│   ├ disco_noises/, TAU-acoustic-sounds/
│   └ CIPIC-HRTF/
├ curated/                             ← Stage 2 결과 (CSV splits)
│   ├ FSD50K/{train,val,test}.csv
│   ├ ESC-50/{train,val,test}.csv
│   └ ...
└ BinauralCuratedDataset/              ← Stage 3 결과 (학습이 사용)
    ├ scaper_fmt/{train,val,test}/{20-class}/*.wav  (symlinks)
    ├ bg_scaper_fmt/{train,val,test}/{class}/
    ├ noise_scaper_fmt/{train,val,test}/{scene}/
    ├ hrtf/CIPIC/{*.sofa, train_hrtf.txt, val_hrtf.txt, test_hrtf.txt}
    └ start_times.csv
```

학습/평가 스크립트는 `--data_dir` 인자로 **`/path/to/data` (BinauralCuratedDataset 의 부모)** 를 받습니다.

다음 단계: [B. 온라인 batch 생성 →](./batch-creation.md)
