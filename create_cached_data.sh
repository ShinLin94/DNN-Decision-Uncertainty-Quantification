#!/bin/bash
#SBATCH --account=b1042
#SBATCH --partition=genomicsguest
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8          # Gives 8 CPU cores for DataLoader num_workers=8
#SBATCH --mem=64G                  # Enough RAM to handle the ~26GB tensor + overhead
#SBATCH --time=02:00:00
#SBATCH --job-name=hpa_create_cached_data
#SBATCH --output=hpa_create_cached_data_%j.log

python -u 08_create_cached_data.py