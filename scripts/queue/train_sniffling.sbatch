#!/bin/bash
#SBATCH --job-name=miso_sniffling
#SBATCH --partition=gpu-l40s
#SBATCH --account=intelligentsystems
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/training_sniffling_%j.log

source /mmfs1/gscratch/intelligentsystems/vidya26/workspace/project/miso/miniconda3/bin/activate miso

cd /mmfs1/gscratch/intelligentsystems/vidya26/workspace/project/miso

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# bash scripts/train/run_tse.sh data/SnifflingOnly miso_sniffling
bash scripts/train/run_tse.sh data/SnifflingOnly miso_sniffling \
    --init_from runs/miso_sniffling/best-epoch=66.ckpt
