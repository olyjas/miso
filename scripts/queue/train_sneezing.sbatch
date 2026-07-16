#!/bin/bash
#SBATCH --job-name=miso_sneezing
#SBATCH --partition=gpu-l40s
#SBATCH --account=intelligentsystems
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/training_sneezing_%j.log


source /mmfs1/gscratch/intelligentsystems/vidya26/workspace/project/miso/miniconda3/bin/activate miso

cd /mmfs1/gscratch/intelligentsystems/vidya26/workspace/project/miso

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

bash scripts/train/run_tse.sh data/SneezingOnly miso_sneezing
