#!/bin/bash
#SBATCH --job-name=codenas
#SBATCH --nodes=1
#SBATCH -c 16
#SBATCH --time=16:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/codenas_predict_%j.out

module load anaconda3/2023.03
conda run -n codenas --no-capture-output python -u current_comp.py