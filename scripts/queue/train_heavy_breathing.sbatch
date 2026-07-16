#!/bin/bash
#SBATCH --job-name=miso_heavy_breathing
#SBATCH --partition=gpu-rtx6k
#SBATCH --account=intelligentsystems
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/training_heavy_breathing_%j.log

source /mmfs1/gscratch/intelligentsystems/vidya26/workspace/project/miso/miniconda3/bin/activate miso

cd /mmfs1/gscratch/intelligentsystems/vidya26/workspace/project/miso

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

bash scripts/train/run_tse.sh data/HeavyBreathingOnly miso_heavy_breathing \
    --init_from runs/miso_heavy_breathing/best-epoch=71.ckpt
