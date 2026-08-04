#!/bin/bash
#SBATCH --account=b1042
#SBATCH --partition=genomicsguest
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --job-name=hpa_train
#SBATCH --output=hpa_create_cached_data_%j.log

python -u 07_create_cached_data.py