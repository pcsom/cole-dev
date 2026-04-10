#!/bin/bash
#SBATCH --job-name=codenas
#SBATCH --nodes=1
#SBATCH -c 12
#SBATCH --time=5:00:00
#SBATCH --mem=48G
#SBATCH -G 1
#SBATCH --constraint="H100|H200"
#SBATCH --output=logs/corrected_nb201_ablation_%j.out

# Accept argument for mode
MODE=${1:-main}

nvidia-smi
module load anaconda3/2023.03
export HF_HOME=/storage/ice-shared/vip-vvk/data/AOT/$USER/huggingface
export HF_TOKEN="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"
conda run -n codenas --no-capture-output python -u paper_ablation_comp.py --mode $MODE