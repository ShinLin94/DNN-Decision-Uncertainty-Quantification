print("Importing things")

from confidence_functions import *

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import os
# import kagglehub

from PIL import Image
import pandas as pd
import numpy as np



print(torch.cuda.is_available())

import socket
print(socket.gethostname())

# use all available cpu nodes
torch.set_num_threads(os.cpu_count())

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Download data if have not
# path = kagglehub.dataset_download("thedrcat/hpa-cell-tiles-sample-balanced-dataset")
# print("Path to dataset files:", path))

print("Reading CSV and preparing data")
# csv_path = os.path.join(path, "cell_df.csv")
# cell_dir = os.path.join(path, "cells"
csv_path = '/home/ykc0662/.cache/kagglehub/datasets/thedrcat/hpa-cell-tiles-sample-balanced-dataset/versions/1/cell_df.csv'
cell_dir = '/home/ykc0662/.cache/kagglehub/datasets/thedrcat/hpa-cell-tiles-sample-balanced-dataset/versions/1/cells'

df = pd.read_csv(csv_path, usecols=['image_id', 'cell_id', 'image_labels'])



transform = transforms.Compose([
    transforms.Resize((224, 224)),  # very low resolution, but faster this way
])

class SimpleImageDataset(Dataset):
    def __init__(self, paths, transform):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        img = self.transform(img)
        img = torch.from_numpy(np.array(img)).permute(2, 0, 1)  # (3, 224, 224), uint8 (reduce mem from float32)
        return img


image_paths = [
    os.path.join(cell_dir, f"{im_id}_{cell_id}.jpg")
    for im_id, cell_id in zip(df['image_id'], df['cell_id'])
]
print(f"Found {len(image_paths)} image paths")


multihot_labels = np.zeros((len(df), 19), dtype=np.uint8)
for i, label_str in enumerate(df['image_labels']):
    for label in label_str.split('|'):
        multihot_labels[i, int(label)] = 1



# Load then transform and save all images to a cached file
cache_path = 'cached_data_224.pt'
if os.path.exists(cache_path):
    print(f"Loading cached data from {cache_path}, (should be fast)")
    cache = torch.load(cache_path)
    all_imgs = cache['images']
    all_labels = cache['labels']
else:
    print(f"Creating cached data at {cache_path}, (this may take a while)")

    dataset = SimpleImageDataset(image_paths, transform)
    # multi-process loader; match num_workers to SLURM cpus-per-task
    loader = DataLoader(dataset, batch_size=5000, shuffle=False, num_workers=8) 

    all_imgs_list = []
    for i, batch in enumerate(loader):
        all_imgs_list.append(batch)
        print(f"Batch {i+1}/{len(loader)} done")
    all_imgs = torch.cat(all_imgs_list, dim=0)
    all_labels = torch.tensor(multihot_labels, dtype=torch.float32)

    torch.save({'images': all_imgs, 'labels': all_labels}, cache_path)
    
print(f"Cached data loaded with {all_imgs.shape[0]} images and {all_labels.shape[0]} labels")