#!/bin/bash
#SBATCH --job-name=miso_tapping_v2
#SBATCH --partition=gpu-rtx6k
#SBATCH --account=intelligentsystems
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:rtx6k:1
#SBATCH --time=48:00:00
#SBATCH --output=logs/tapping_v2/training_tapping_v2_%j.log
#SBATCH --exclude=g3108


source /mmfs1/gscratch/intelligentsystems/vidya26/workspace/project/miso/miniconda3/bin/activate miso

cd /mmfs1/gscratch/intelligentsystems/vidya26/workspace/project/miso

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

bash scripts/train/run_tse.sh data/TappingOnly miso_tapping_v2
