#!/bin/bash
#SBATCH --job-name=miso_cough
#SBATCH --partition=ckpt
#SBATCH --account=demo-ckpt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --output=logs/training_cough_%j.log

set -euo pipefail
cd /mmfs1/gscratch/scrubbed/jaszhang/miso/miso
mkdir -p logs

# Frankenstein Python env: coenv's stdlib + miso conda env's binary/packages
# (the miso conda env's own stdlib was purged from the scrubbed filesystem)
export MISO=/mmfs1/gscratch/scrubbed/jaszhang/miniconda3/envs/miso
export COENV=/mmfs1/sw/contrib/coenv-src/python/3.11.9
export LD_LIBRARY_PATH=$MISO/lib:$COENV/lib:${LD_LIBRARY_PATH:-}
export PYTHONHOME=$COENV
# pylib_patch/packaging: the conda env's own copy had its .py source files purged
# (scrubbed-FS cleanup); this is a working copy of the same pure-Python package.
export PYTHONPATH=$(pwd)/pylib_patch:$MISO/lib/python3.11/lib-dynload:$MISO/lib/python3.11/site-packages
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PYTHON=$MISO/bin/python3.11

echo "=== Host: $(hostname) ==="
$PYTHON -c "import torch; print('torch', torch.__version__, 'cuda:', torch.cuda.is_available())"

echo "=== Starting cough training ==="
$PYTHON -m src.tse.train \
    --config configs/tse/miso_cough.yaml \
    --data_dir data/CoughOnly

echo "=== Training finished ==="
