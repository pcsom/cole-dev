#!/bin/bash
#SBATCH --job-name=codenas
#SBATCH --nodes=1
#SBATCH -c 16
#SBATCH --time=10:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/codenas_robust_nb201_%j.out

nvidia-smi
module load anaconda3/2023.03
export HF_HOME=/storage/ice-shared/vip-vvk/data/AOT/$USER/huggingface
export HF_TOKEN="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"
conda run -n codenas --no-capture-output python -u robust_nas_comp.py