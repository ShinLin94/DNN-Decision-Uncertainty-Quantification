print("Importing things")

from confidence_functions import *

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.models as models

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


print("Reading CSV and preparing data")

csv_path = '/home/ykc0662/.cache/kagglehub/datasets/thedrcat/hpa-cell-tiles-sample-balanced-dataset/versions/1/cell_df.csv'
cell_dir = '/home/ykc0662/.cache/kagglehub/datasets/thedrcat/hpa-cell-tiles-sample-balanced-dataset/versions/1/cells'

df = pd.read_csv(csv_path, usecols=['image_id', 'cell_id', 'image_labels'])



transform = transforms.Compose([
    transforms.Resize((50, 50)),  # very low resolution, but faster this way
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


image_paths = [
    os.path.join(cell_dir, f"{im_id}_{cell_id}.jpg")
    for im_id, cell_id in zip(df['image_id'], df['cell_id'])
]
print(f"Found {len(image_paths)} image paths")


multihot_labels = np.zeros((len(df), 19), dtype=np.uint8)
for i, label_str in enumerate(df['image_labels']):
    for label in label_str.split('|'):
        multihot_labels[i, int(label)] = 1



cache_path = 'cached_data.pt'
if os.path.exists(cache_path):
    print(f"Loading cached data from {cache_path}")
    cache = torch.load(cache_path)
    all_imgs = cache['images']
    all_labels = cache['labels']
else:
    print(f"Creating cached data at {cache_path}")
    all_imgs = torch.stack([transform(Image.open(p).convert('RGB')) for p in image_paths])
    all_labels = torch.tensor(multihot_labels, dtype=torch.float32)
    torch.save({'images': all_imgs, 'labels': all_labels}, cache_path)
    
print(f"Cached data loaded with {all_imgs.shape[0]} images and {all_labels.shape[0]} labels")
