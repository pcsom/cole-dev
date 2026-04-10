#!/bin/bash
#SBATCH --job-name=general_clean
#SBATCH -c 8
#SBATCH --time=6:00:00
#SBATCH --mem=8g
#SBATCH --output=logs/general_%j.out

ENVS_TO_DELETE=("code2prompt" "cv_proj3" "cv_proj4" "cv_proj5" "cv_proj6" "dl1" "dl2" "dl3" "dl4" "dmytro7" "finaldl" "hack" "hw4" "naslib39" "nasganet" "piml")

# Loop through the list and remove each environment
for env_name in "${ENVS_TO_DELETE[@]}"; do
  echo "Attempting to remove Conda environment: $env_name"
  # Use -n for the name, --all to remove all packages in the env, and -y to skip confirmation
  conda remove -n "$env_name" --all -y
  if [ $? -eq 0 ]; then
    echo "Successfully removed $env_name"
  else
    echo "Failed to remove $env_name, or it did not exist."
  fi
done

echo "Environment removal process complete."