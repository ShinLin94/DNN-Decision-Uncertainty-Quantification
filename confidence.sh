#!/bin/bash
#SBATCH --account=b1042
#SBATCH --partition=genomicsguest
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4          # Gives 8 CPU cores for DataLoader num_workers=8          
#SBATCH --mem=200G                 
#SBATCH --time=02:00:00
#SBATCH --job-name=hpa_visualize_confidence
#SBATCH --output=hpa_visualize_confidence_%j.log

python -u 11_visualizing_confidence.py