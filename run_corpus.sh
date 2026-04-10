#!/bin/bash
#SBATCH --job-name=codenas
#SBATCH -c 10
#SBATCH --time=5:00:00
#SBATCH --mem=64G
#SBATCH --output=logs/do_embed_%j.out

module load anaconda3/2023.03
conda run -n codenas --no-capture-output python -u do_embed.py