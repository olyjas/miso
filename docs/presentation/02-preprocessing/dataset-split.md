# 02-2. Offline Split — Datasets → train/val/test.csv

[← Preprocessing Overview](./overview.md)

This stage is a one-time job that **downloads several public datasets and produces per-class train/val/test split CSVs along with scaper-format symlink directories**.

---

## Entry Point

```bash
# Recommended: one-line invocation
bash scripts/setup_dataset.sh --output_dir /path/to/data

# Underlying Python entry point
python data/setup_data.py --output_dir <path> --stage all \
                          [--datasets fsd50k,esc50,disco,cipic,musdb18,tau] \
                          [--manual_dir <path>] \
                          [--reference_dir <path>]
```

Overall flow (`data/pipeline/__init__.py:13`):

```
run(output_dir, stage='all', datasets=None, ...)
  │ random.seed(0); np.random.seed(0)   ← seeds for reproducibility
  │
  ├─→ Stage 1: download
  │     data/pipeline/download.py:run_download
  │     calls each source's BaseSource.download()
  │
  ├─→ Stage 2: collect
  │     data/pipeline/collect.py:run_collect
  │     calls each source's BaseSource.collect() → emits train/val/test.csv
  │
  └─→ Stage 3: prepare
        data/pipeline/prepare.py:run_prepare
        reads CSVs → scaper_fmt/ symlinks + HRTF split + start_times.csv
```

---

## Stage 1: download

`data/pipeline/download.py:run_download` invokes each source's `download(raw_dir)` method.

| Dataset | Download method | Location |
|---|---|---|
| FSD50K | Auto-downloaded from HF mirror `Fhrozen/FSD50k` (or manual via Zenodo) | `raw/FSD50K/` |
| ESC-50 | Auto-downloaded GitHub zip | `raw/ESC-50/` |
| DISCO | Auto from Zenodo (4019030) | `raw/disco_noises/` |
| CIPIC | HF dataset `ooshyun/...` or UC Davis | `raw/CIPIC-HRTF/` |
| **musdb18** | Manual download required (license) | `raw/musdb18/` |
| **TAU-2019** | Manual download (Zenodo 3063822, 11 audio + meta) | `raw/TAU-acoustic-sounds/` |

For datasets that require manual downloads, point to their location with `--manual_dir <path>`.

---

## Stage 2: collect — Per-Source CSV Generation

`run_collect` in `data/pipeline/collect.py:6` calls into the core `BaseSource` interface (`data/pipeline/sources/base.py:8`):

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

### 2-1. Prefer Reference CSVs When Provided

`base.py:26` `try_use_reference_csvs()` — if `--reference_dir` is supplied, it copies pre-validated CSVs verbatim (downloaded from HF `ooshyun/fine_grained_soundscape_control/reference_splits/`). This is the recommended path for guaranteeing identical splits across runs.

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

### 2-2. FSD50K Split Logic (Example)

`data/pipeline/sources/fsd50k.py:124` (assuming the Zenodo layout):

```
1. Load pp_pnp_ratings_FSD50K.json → quality filter
   → keep only samples that have ≥2 positives and 0 negatives/uncertain per label

2. collection_dev.csv → dev_curated (single-label only)
   collection_eval.csv → test_samples

3. Stratified 90:10 split of dev_curated by label:
   sklearn.model_selection.train_test_split(test_size=0.1, random_state=42)
   → train_samples, val_samples

4. Keep only classes that appear in all three splits (intersection of common_labels)

5. Write CSVs: train.csv / val.csv / test.csv
   columns: [fname, label, id]
```

Refer to `data/pipeline/sources/{esc50,disco,musdb18,tau,cipic}.py` for the split rules of the other sources.

### 2-3. CSV Output Format

`base.py:52` `_write_csvs()`:

```python
out_dir = curated_dir / self.name  # e.g. curated/FSD50K/
for split, df in [("train", train), ("val", val), ("test", test)]:
    df.to_csv(out_dir / f"{split}.csv", index=False)
```

CSV columns vary by source, but the common keys are:

| Column | Meaning |
|---|---|
| `fname` | Path relative to the dataset root (e.g. `FSD50K.dev_audio/12345.wav`) |
| `label` | AudioSet class name (e.g. `Dog`, `Bark`) |
| `id` | AudioSet ID (e.g. `/m/0bt9lr`) |

---

## Stage 3: prepare — Scaper-Format Symlinks

`run_prepare` at `data/pipeline/prepare.py:252` proceeds in four substeps:

### 3-1. Build the id2classname Mapping

`prepare.py:30` `build_id2classname()`:

```python
class_map: dict[str, list[str]] = yaml.safe_load(open("data/class_map.yaml"))
# e.g. {"dog": ["Dog"], "cat": ["Cat", "Domestic cat"], ...}

id2classname = {}
for class_name, elements in class_map.items():
    for element_name in elements:
        node_id = ontology.get_id_from_name(element_name)
        # Map the entire subtree to the same class_name
        for cid in ontology.get_subtree(node_id):
            id2classname[cid] = class_name
```

→ Result: AudioSet IDs are mapped to one of the 20 classes (subtree included).

### 3-2. Create Scaper-Format Symlinks

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
                # background (excludes Music/Human voice and any FG overlap)
                out_path = bg_scaper_fmt/{split}/{label}/
            else:
                continue

            # Create a relative symlink
            os.symlink("../../../{dataset_name}/{fname}", out_path/fname)

            # Silence-trim metadata
            start, first_silence, end = trim_silence(audio)
            all_samples.append({fname, start_sample, end_sample, first_silence})
```

### 3-3. HRTF Split

`prepare.py:191` `prepare_hrtf()`:

```
1. Locate CIPIC SOFA files under raw/CIPIC-HRTF/ or raw/cipic-hrtf-database/
2. Copy them to <output_dir>/hrtf/CIPIC/
3. random.shuffle(copied) — uses random.seed(0) for reproducibility
4. 80:10:10 split → train_hrtf.txt / val_hrtf.txt / test_hrtf.txt
```

### 3-4. Write start_times.csv

Consolidates silence-trim metadata in a single file:

```
output_dir/start_times.csv
  columns: [fname, start_sample, end_sample, first_silence]
```

`SoundscapeDataset.sample_snippet()` consults this file at training time to avoid silent regions.

---

## Final Directory Layout

```
/path/to/data/
├ raw/                                 ← Stage 1 output (raw downloads)
│   ├ FSD50K/, ESC-50/, musdb18/
│   ├ disco_noises/, TAU-acoustic-sounds/
│   └ CIPIC-HRTF/
├ curated/                             ← Stage 2 output (CSV splits)
│   ├ FSD50K/{train,val,test}.csv
│   ├ ESC-50/{train,val,test}.csv
│   └ ...
└ BinauralCuratedDataset/              ← Stage 3 output (consumed by training)
    ├ scaper_fmt/{train,val,test}/{20-class}/*.wav  (symlinks)
    ├ bg_scaper_fmt/{train,val,test}/{class}/
    ├ noise_scaper_fmt/{train,val,test}/{scene}/
    ├ hrtf/CIPIC/{*.sofa, train_hrtf.txt, val_hrtf.txt, test_hrtf.txt}
    └ start_times.csv
```

Training and evaluation scripts accept `--data_dir` pointing to **`/path/to/data` (the parent of BinauralCuratedDataset)**.

Next: [B. Online batch creation →](./batch-creation.md)
