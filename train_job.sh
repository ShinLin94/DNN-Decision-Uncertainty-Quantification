#!/bin/bash
#SBATCH --account=b1042
#SBATCH --partition=genomics-gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4        # 4 CPUs is plenty since data is cached in RAM
#SBATCH --mem=64G                # 24GB for dataset + Python overhead
#SBATCH --time=06:00:00
#SBATCH --job-name=hpa_train_gpu
#SBATCH --output=hpa_train_gpu_%j.log

source /gpfs/projects/b1042/AmaralLab/shin/DNN-Decision-Uncertainty-Quantification/myenv/bin/activate
python -u 09_single_cell_training.py