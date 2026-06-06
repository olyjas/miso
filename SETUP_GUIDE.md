# Hyak Setup Guide — MISO Project
**For new contributors pulling this repo on Hyak**

---

## Step 1: Clone the repo

```bash
git clone <your-repo-url>
cd miso
```

---

## Step 2: Set up Python environment (CRITICAL — read this carefully)

**Do NOT use Hyak's system Python or coenv modules.** They are missing a required system library (`_ctypes`) and will crash with:
```
ModuleNotFoundError: No module named '_ctypes'
```

You need your own Miniconda installation.

### Install Miniconda (if you don't have it)

```bash
cd /mmfs1/gscratch/scrubbed/<your-uwnetid>
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p ./miniconda3
source ./miniconda3/bin/activate
conda tos accept
```

### Create the miso environment

```bash
conda create -n miso python=3.11 -y
source /mmfs1/gscratch/scrubbed/<your-uwnetid>/miniconda3/bin/activate miso
cd /path/to/miso/repo
pip install -r requirements.txt
```

### Activate the environment every session

```bash
source /mmfs1/gscratch/scrubbed/<your-uwnetid>/miniconda3/bin/activate miso
```

---

## Step 3: Point to the dataset

The dataset already lives at:
```
/mmfs1/gscratch/scrubbed/jaszhang/miso/miso/data/MisoDataset/
```

You can symlink it into your repo instead of copying (saves disk space):
```bash
cd /path/to/your/miso/repo
ln -s /mmfs1/gscratch/scrubbed/jaszhang/miso/miso/data/MisoDataset data/MisoDataset
```

---

## Step 4: Get a GPU node

### Find available partitions first
```bash
sinfo --format="%P %a %l %G"
```

Look for partitions with `gpu` in the GRES column. Common ones:
- `ckpt` — works for everyone, but jobs can be preempted (killed) if a priority user needs the GPU
- `gpu-a40` / `gpu-rtx6k` / club lab partitions — check what your account has access to

### Request a GPU node
```bash
srun --partition=ckpt --account=<your-account> \
     --nodes=1 --ntasks=1 --cpus-per-task=4 \
     --mem=16G --gres=gpu:1 --time=4:00:00 \
     --pty /bin/bash
```

Replace `<your-account>` with your SLURM account. To find yours:
```bash
sacctmgr show associations user=$USER format=account
```

If you have club lab access, replace `ckpt` with your club partition name — you get priority and won't be preempted.

---

## Step 5: Run training inside tmux (so it survives laptop sleep)

```bash
# Start a tmux session
tmux new -s training

# Activate environment
source /mmfs1/gscratch/scrubbed/<your-uwnetid>/miniconda3/bin/activate miso
cd /path/to/miso/repo

# Run training
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
bash scripts/train/run_tse.sh data/MisoDataset miso
```

**Detach from tmux** (training keeps running): `Ctrl+B` then `D`

**Reattach later:** `tmux attach -t training`

---

## Step 6: Resume from existing checkpoint (optional)

If a checkpoint already exists and you want to continue from it:
```bash
bash scripts/train/run_tse.sh data/MisoDataset miso \
  --resume_from runs/miso/best-epoch=XX.ckpt
```

---

## Step 5b: Set up rclone for Google Drive (only needed if re-downloading dataset)

### Install rclone
```bash
curl -O https://downloads.rclone.org/rclone-current-linux-amd64.zip
unzip rclone-current-linux-amd64.zip
mkdir -p ~/.local/bin
mv rclone-*-linux-amd64/rclone ~/.local/bin/
export PATH="$HOME/.local/bin:$PATH"
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
```

### Configure Google Drive
```bash
rclone config
```
When prompted:
1. Press `n` for new remote
2. Name it `gdrive`
3. Choose `drive` (Google Drive) from the list
4. Leave client_id and client_secret blank (press Enter)
5. Choose scope `1` (full access)
6. Press Enter for root folder and service account
7. When asked "Use auto config?" press `n` (Hyak has no browser)
8. Copy the long URL it gives you, open it on your **local laptop**
9. Log in to Google, approve access, copy the verification code back into Hyak terminal
10. Press `n` for team drive, then `y` to confirm

### Test it works
```bash
rclone ls "gdrive:/CSE 481 Project/MISO_Dataset" --drive-shared-with-me
```

### Download trigger sounds (example: chewing)
```bash
rclone copy "gdrive:/CSE 481 Project/MISO_Dataset/clean_triggers/chewing" \
  data/MisoDataset/raw/chewing \
  --drive-shared-with-me --progress
```

### Download background noise
```bash
rclone copy "gdrive:/CSE 481 Project/MISO_Dataset/background" \
  data/MisoDataset/raw/background \
  --drive-shared-with-me --progress
```

### Split files into train/val (80/20)
```bash
CLASS="chewing"  # change for each class
SRC="data/MisoDataset/raw/$CLASS"
TRAIN="data/MisoDataset/scaper_fmt/train/$CLASS"
VAL="data/MisoDataset/scaper_fmt/val/$CLASS"
mkdir -p "$TRAIN" "$VAL"

files=("$SRC"/*.wav)
total=${#files[@]}
n_train=$(( total * 80 / 100 ))

for i in "${!files[@]}"; do
  if [ "$i" -lt "$n_train" ]; then
    cp "${files[$i]}" "$TRAIN/"
  else
    cp "${files[$i]}" "$VAL/"
  fi
done
echo "$CLASS: train=$(ls $TRAIN | wc -l)  val=$(ls $VAL | wc -l)"
```

Run this for each class: `chewing`, `cough`, `drinking`, `heavy_breathing`, `sneeze`, `sniffling`, `tapping`

### Split background noise
```bash
SRC="data/MisoDataset/raw/background"
TRAIN="data/MisoDataset/noise_scaper_fmt/train/ambient"
VAL="data/MisoDataset/noise_scaper_fmt/val/ambient"
mkdir -p "$TRAIN" "$VAL"

files=("$SRC"/*.wav)
total=${#files[@]}
n_train=$(( total * 80 / 100 ))

for i in "${!files[@]}"; do
  if [ "$i" -lt "$n_train" ]; then
    cp "${files[$i]}" "$TRAIN/"
  else
    cp "${files[$i]}" "$VAL/"
  fi
done
echo "background: train=$(ls $TRAIN | wc -l)  val=$(ls $VAL | wc -l)"
```

### Copy HRTF files (symlink from Jasmine's dataset)
```bash
mkdir -p data/MisoDataset/hrtf/CIPIC
ln -s /mmfs1/gscratch/scrubbed/jaszhang/miso/miso/data/BinauralCuratedDataset/hrtf/CIPIC/*.sofa \
  data/MisoDataset/hrtf/CIPIC/
cp /mmfs1/gscratch/scrubbed/jaszhang/miso/miso/data/MisoDataset/hrtf/CIPIC/train_hrtf.txt \
  data/MisoDataset/hrtf/CIPIC/
cp /mmfs1/gscratch/scrubbed/jaszhang/miso/miso/data/MisoDataset/hrtf/CIPIC/val_hrtf.txt \
  data/MisoDataset/hrtf/CIPIC/
```

---

## Common Errors and Fixes

### `ModuleNotFoundError: No module named '_ctypes'`
**Cause:** Using system Python or coenv module  
**Fix:** Install your own Miniconda (see Step 2)

### `ModuleNotFoundError: No module named '_ctypes'` even with conda
**Cause:** You activated the wrong environment  
**Fix:** Run `which python` — it should show your miniconda path, not `/usr/bin/python`

### `CUDA out of memory`
**Cause:** Batch size too large for the GPU  
**Fix:** In `configs/tse/miso.yaml`, set `batch_size: 2`  
Also add before running:
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### `srun: error: invalid partition`
**Cause:** Partition name doesn't exist or you don't have access  
**Fix:** Run `sinfo --format="%P %a %l %G"` to see available partitions

### `ValueError: Not enough FG sounds ... Expected >= 7, found X`
**Cause:** The fg_dir has wrong number of class folders  
**Fix:** Check `data/MisoDataset/scaper_fmt/train/` has exactly 7 folders:
```
chewing  cough  drinking  heavy_breathing  sneeze  sniffling  tapping
```
No empty folders, no extra folders.

### `Failed to sample valid audio from .../tapping`
**Cause:** Tapping val files are not in .wav format  
**Fix:** Convert them to .wav — this is a known pending issue. It causes warnings but does not crash training.

### `conda: bad interpreter: Permission denied`
**Cause:** Using someone else's system miniconda  
**Fix:** Install your own (see Step 2)

### `conda tos` error on first run
**Fix:**
```bash
conda tos accept
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
```

### `rclone: command not found`
**Cause:** rclone not installed and not in PATH  
**Fix:**
```bash
curl -O https://downloads.rclone.org/rclone-current-linux-amd64.zip
unzip rclone-current-linux-amd64.zip
mkdir -p ~/.local/bin
mv rclone-*-linux-amd64/rclone ~/.local/bin/
export PATH="$HOME/.local/bin:$PATH"
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
```

### Training stops when laptop goes to sleep
**Fix:** Always run training inside `tmux` (see Step 5)

### Job gets preempted (killed mid-training)
**Cause:** Using `ckpt` partition — another user took the GPU  
**Fix:** Restart with `--resume_from` pointing to the last saved checkpoint:
```bash
bash scripts/train/run_tse.sh data/MisoDataset miso \
  --resume_from runs/miso/best-epoch=XX.ckpt
```
Use a club lab partition if available — jobs there are not preempted.

---

## Run inference on a trained model

```bash
python infer.py \
  --ckpt runs/miso/best-epoch=XX.ckpt \
  --input my_audio.wav \
  --trigger chewing \
  --output clean_output.wav
```

Valid trigger names: `chewing`, `cough`, `drinking`, `heavy_breathing`, `sneeze`, `sniffling`, `tapping`

You can remove multiple at once:
```bash
python infer.py --ckpt runs/miso/best-epoch=XX.ckpt \
  --input my_audio.wav \
  --trigger chewing tapping \
  --output clean_output.wav
```

---

## Key file locations

```
configs/tse/miso.yaml              ← training config (edit this to tune training)
src/datasets/soundscape_dataset.py ← dataset — ground truth = clean background
src/tse/train.py                   ← training entry point
infer.py                           ← inference script
runs/miso/                         ← saved checkpoints go here
data/MisoDataset/                  ← dataset
  scaper_fmt/train/{class}/        ← trigger audio clips
  noise_scaper_fmt/train/ambient/  ← background noise clips
  hrtf/CIPIC/                      ← HRTF files for binaural simulation
```

---

## Check training progress

From any login node (no GPU needed):
```bash
ls -lh runs/miso/
```

The filename tells you the best epoch so far. The timestamp tells you when it was last updated.
