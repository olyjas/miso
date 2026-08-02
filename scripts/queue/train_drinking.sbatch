#!/bin/bash
#SBATCH --job-name=miso_drinking
#SBATCH --partition=gpu-a40
#SBATCH --account=intelligentsystems
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:a40:1
#SBATCH --time=48:00:00
#SBATCH --output=logs/training_drinking_and_slurping%j.log
#SBATCH --exclude=g3108


source /mmfs1/gscratch/intelligentsystems/vidya26/workspace/project/miso/miniconda3/bin/activate miso

cd /mmfs1/gscratch/intelligentsystems/vidya26/workspace/project/miso

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

bash scripts/train/run_tse.sh data/DrinkingAndSlurpingOnly miso_drinking
