#!/bin/bash
#SBATCH --account=b1042
#SBATCH --partition=genomicsguest
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4         
#SBATCH --mem=200G                 
#SBATCH --time=00:30:00
#SBATCH --job-name=hpa_alpha
#SBATCH --output=hpa_alpha%j.log

python -u alpha.py