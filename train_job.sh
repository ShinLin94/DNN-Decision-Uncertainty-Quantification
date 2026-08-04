#!/bin/bash
#SBATCH --account=b1042
#SBATCH --partition=genomics-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --job-name=hpa_train_gpu
#SBATCH --output=hpa_train_gpu_%j.log

source /gpfs/projects/b1042/AmaralLab/shin/DNN-Decision-Uncertainty-Quantification/myenv/bin/activate
python -u 07_single_cell_training.py