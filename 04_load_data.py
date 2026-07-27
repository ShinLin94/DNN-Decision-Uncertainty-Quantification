import os
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi

# Load train.csv and take a small random sample
df = pd.read_csv('/gpfs/projects/b1042/AmaralLab/shin/DNN-Decision-Uncertainty-Quantification/train.csv')
subset = df.sample(n=100, random_state=42)

# Save the subset CSV
subset.to_csv('/gpfs/projects/b1042/AmaralLab/shin/DNN-Decision-Uncertainty-Quantification/train_subset.csv', index=False)
print(f"Created subset with {len(subset)} rows.")

# Load the subset CSV
subset_df = pd.read_csv('/gpfs/projects/b1042/AmaralLab/shin/DNN-Decision-Uncertainty-Quantification/train_subset.csv')

# Authenticate Kaggle API
api = KaggleApi()
api.authenticate()

target_dir = '/gpfs/projects/b1042/AmaralLab/shin/DNN-Decision-Uncertainty-Quantification/data/train_subset_images'
os.makedirs(target_dir, exist_ok=True)

# Download four channel files for each sample ID in the subset
colors = ['red', 'blue', 'green', 'yellow']
competition = 'hpa-single-cell-image-classification'

print("Starting subset download...")
for image_id in subset_df['ID']:
    for color in colors:
        filename = f"train/{image_id}_{color}.png"
        try:
            api.competition_download_file(competition, filename, path=target_dir)
        except Exception as e:
            print(f"Failed to download {filename}: {e}")

print("Subset image download complete!")