#!/bin/bash
#SBATCH --job-name=miso_infer
#SBATCH --partition=ckpt
#SBATCH --account=demo-ckpt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=0:30:00
#SBATCH --output=infer_output.log
#SBATCH --nodelist=z3001
#SBATCH --export=ALL

# Skip conda activation entirely — it calls broken base Python 3.13
# Set up the miso env directly
MISO_ENV=/mmfs1/gscratch/scrubbed/jaszhang/miniconda3/envs/miso
export PYTHONHOME=$MISO_ENV
export PATH=$MISO_ENV/bin:$PATH
export LD_LIBRARY_PATH=$MISO_ENV/lib:$LD_LIBRARY_PATH

PYTHON=$MISO_ENV/bin/python3.11

echo "=== Python test ==="
$PYTHON -c "import sys; print('Python OK:', sys.version)" || { echo "PYTHON STILL BROKEN"; exit 1; }

echo "=== Mixing audio ==="
$PYTHON - << 'PYEOF'
import soundfile as sf
import numpy as np
bg, sr = sf.read('data/MisoDataset/noise_scaper_fmt/val/ambient/bg_long_3735000.wav')
fg, _ = sf.read('data/MisoDataset/scaper_fmt/val/tapping/tap_long_690000.wav')
n = min(len(bg), len(fg))
bg, fg = bg[:n], fg[:n]
if bg.ndim > 1: bg = bg[:, 0]
if fg.ndim > 1: fg = fg[:, 0]
mixed = bg + fg * 0.7
mixed = mixed / np.abs(mixed).max()
sf.write('test_tapping.wav', mixed, sr)
print('Mix done')
PYEOF

echo "=== Running inference ==="
$PYTHON infer.py \
  --ckpt runs/miso_tapping/best-epoch=49.ckpt \
  --input test_tapping.wav \
  --trigger tapping \
  --output clean_output.wav

echo "=== Inference done ==="
