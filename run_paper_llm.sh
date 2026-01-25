#!/bin/bash
#SBATCH --job-name=codenas
#SBATCH --nodes=1
#SBATCH -c 12
#SBATCH --time=16:00:00
#SBATCH --mem=96G
#SBATCH --gres=gpu:h200
#SBATCH --output=logs/corrected_nb201_%j.out

# Accept two arguments: model1 and model2
MODEL1=${1:-codellama_python_7b}
MODEL2=${2:-modernbert_large}

nvidia-smi
module load anaconda3/2023.03
export HF_HOME=/storage/ice-shared/vip-vvk/data/AOT/$USER/huggingface
export HF_TOKEN="redacted"
conda run -n codenas --no-capture-output python -u paper_llm_comp.py --model1 $MODEL1 --model2 $MODEL2