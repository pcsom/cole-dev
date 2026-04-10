#!/bin/bash
#SBATCH --job-name=codenas_paper
#SBATCH --nodes=1
#SBATCH -c 12
#SBATCH --time=3:00:00
#SBATCH --mem=48G
#SBATCH -G 1
#SBATCH --constraint="H100|H200"
#SBATCH --output=logs/corrected_nb101_%j.out

nvidia-smi
module load anaconda3/2023.03
export HF_HOME=/storage/ice-shared/vip-vvk/data/AOT/$USER/huggingface
export HF_TOKEN="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"
conda run -n codenas --no-capture-output python -u concat_comp.py