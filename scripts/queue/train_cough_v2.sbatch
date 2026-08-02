#!/bin/bash
#SBATCH --job-name=miso_cough_v2
#SBATCH --partition=gpu-a40
#SBATCH --account=intelligentsystems
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:a40:1
#SBATCH --time=48:00:00
#SBATCH --output=logs/cough_v3/training_cough_v3_%j.log
#SBATCH --exclude=g3108


source /mmfs1/gscratch/intelligentsystems/vidya26/workspace/project/miso/miniconda3/bin/activate miso

cd /mmfs1/gscratch/intelligentsystems/vidya26/workspace/project/miso

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

bash scripts/train/run_tse.sh data/CoughOnly miso_cough_v2
