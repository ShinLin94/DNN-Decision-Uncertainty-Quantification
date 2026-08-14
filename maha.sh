#!/bin/bash
#SBATCH --account=b1042
#SBATCH --partition=genomicsguest
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1          # Gives 8 CPU cores for DataLoader num_workers=8          
#SBATCH --mem=200G                 
#SBATCH --time=00:30:00
#SBATCH --job-name=hpa_maha
#SBATCH --output=hpa_maha%j.log

python -u maha.py