#!/usr/bin/env python
# coding: utf-8


# In[3]:
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

from skmultilearn.model_selection import iterative_train_test_split
import matplotlib.pyplot as plt


# In[1]:
print(torch.cuda.is_available())

import socket
print(socket.gethostname())

# use all available cpu nodes
torch.set_num_threads(os.cpu_count())

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# change matplotlib to non-interactive backend, safe for compute nodes
import matplotlib
matplotlib.use('Agg')

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


# In[ ]:


# Download data if have not
# path = kagglehub.dataset_download("thedrcat/hpa-cell-tiles-sample-balanced-dataset")

# print("Path to dataset files:", path)


# In[ ]:


# csv_path = os.path.join(path, "cell_df.csv")
# cell_dir = os.path.join(path, "cells")
csv_path = '/home/ykc0662/.cache/kagglehub/datasets/thedrcat/hpa-cell-tiles-sample-balanced-dataset/versions/1/cell_df.csv'
cell_dir = '/home/ykc0662/.cache/kagglehub/datasets/thedrcat/hpa-cell-tiles-sample-balanced-dataset/versions/1/cells'

df = pd.read_csv(csv_path, usecols=['image_id', 'cell_id', 'image_labels'])


# ## Checking the data

# In[ ]:


df['image_labels'].value_counts()
df['image_labels'].unique()

print(f"Amount of cells/images: {len(df)}")
print(f"Amount of unique labels: {df['image_labels'].unique().shape}")

label_counts = df['image_labels'].value_counts()
print(f"Min: {label_counts.min()}")
print(f"Max: {label_counts.max()}")
print(f"Median: {label_counts.median()}")


# In[ ]:


single_label_counts = pd.Series(
    [df['image_labels'].value_counts().get(str(i), 0) for i in range(19)],
    index=range(19)
)

plt.figure(figsize=(8, 2))
plt.bar(single_label_counts.index, single_label_counts.values, color='skyblue')
plt.xlabel('Label')
plt.ylabel('Count - log scale')
plt.yscale('log') 
plt.title('Counts for labels 0 through 18')
plt.xticks(range(19))
plt.savefig('plots/label_counts.png')


# In[ ]:


label_counts = df['image_labels'].value_counts()
split_labels = label_counts.index.str.split('|')
# print(split_labels)

weights = np.repeat(label_counts.values, [len(lbl) for lbl in split_labels])
class_ids = np.concatenate(split_labels).astype(int)
# print(weights)

class_appear_count = np.bincount(class_ids, weights=weights, minlength=19)
class_appear_counts = pd.Series(class_appear_count, index=range(19))


plt.figure(figsize=(8, 2))
plt.bar(class_appear_counts.index, class_appear_counts.values, color='skyblue')
plt.xlabel('Label')
plt.ylabel('Appear count - log scale')
plt.yscale('log') 
plt.title('Appear counts for labels 0 through 18')
plt.xticks(range(19))
plt.savefig('plots/class_appear_counts.png')


# In[ ]:


print(f"cell count with label==11: {df['image_labels'].value_counts().get('11', 0)}")
print(f"cell count with label==1|2: {df['image_labels'].value_counts().get('1|2', 0)}")
print(f"appear count of class 18: {class_appear_counts.values[18]}")
print(f"appear count of class 11: {class_appear_counts.values[11]}")


# In[ ]:


multihot_labels = np.zeros((len(df), 19), dtype=np.uint8)
for i, label_str in enumerate(df['image_labels']):
    for label in label_str.split('|'):
        multihot_labels[i, int(label)] = 1


# In[ ]:


# Load all images from cached file
cache_path = 'cached_data_224.pt'
if os.path.exists(cache_path):
    print(f"Loading cached data from {cache_path}, (should be fast)")
    cache = torch.load(cache_path)
    all_imgs = cache['images']
    all_labels = cache['labels']
else:
    print(f"ERROR! Cached data not found at {cache_path}")
    raise FileNotFoundError(f"Cached data not found at {cache_path}")
    
print(f"Cached data loaded with {all_imgs.shape[0]} images and {all_labels.shape[0]} labels")

# In[ ]:


print("Splitting data into train and test sets (10% test size: ~17.4k samples)")
X = np.arange(len(df)).reshape(-1, 1)  # just indices of images
y = np.array(multihot_labels)  # (174000, 19)

# dataset is highly imbalance, but we are just going to use 
# iterative_train_test_split from scikit learn used for multi-label data
X_train, y_train, X_test, y_test = iterative_train_test_split(X, y, test_size=0.1)

train_indices = X_train.flatten()
test_indices = X_test.flatten()

# save train and test indices to disk for reproducibility
torch.save({'train_indices': train_indices, 'test_indices': test_indices}, 'train_test_indices.pt')

train_imgs, train_labels = all_imgs[train_indices], all_labels[train_indices]
test_imgs, test_labels = all_imgs[test_indices], all_labels[test_indices]

print(f"Train set: {train_imgs.shape[0]} images, {train_labels.shape[0]} labels")
print(f"Test set: {test_imgs.shape[0]} images, {test_labels.shape[0]} labels")


# In[ ]:

# find mean and std of the training set for normalization
NORM_MEAN, NORM_STD = compute_dataset_stats(train_imgs, already_scaled_0_1=False)
print(f"Computed mean: {NORM_MEAN}, std: {NORM_STD} for normalization of training set")

# In[ ]:


class CachedCellDataset(Dataset):
    def __init__(self, images, labels, mean=NORM_MEAN, std=NORM_STD):
        self.images = images
        self.labels = labels
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx].float() / 255.0
        img = (img - self.mean) / self.std   # <-- normalization added here
        return img, self.labels[idx]


# In[ ]:

print("Create dataset objects")
# create Dataset object using the CellDataset class
train_dataset = CachedCellDataset(train_imgs, train_labels)
test_dataset = CachedCellDataset(test_imgs, test_labels)

# pass object into DataLoader (Make batch_size higher if you are using GPU for faster speed)
g = torch.Generator()
g.manual_seed(SEED)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, generator=g)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=True, generator=g)


# ## Train Model

# In[ ]:


def train_model(n_epochs, train_loader, test_loader, pos_weight, threshold=0.5, cats=100, output_dim=19):   
    
    # use pretrained resnet18 model
    model = models.resnet18(weights=None)

    # # freeze everything except the last residual block and the final FC layer
    # for name, param in model.named_parameters():
    #     if not (name.startswith('layer4') or name.startswith('fc')):
    #         param.requires_grad = False

    # change output layer to have 19 logits
    model.fc = nn.Linear(512, output_dim)
    model.to(device)
    
    # criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=5e-5)
    # optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)

    X, y = next(iter(test_loader))
    X, y = X.to(device), y.to(device) 
    cat_all = torch.zeros(len(X), output_dim).to(device)
    loss_history = []

    # start calculating cat_all for the last 100 iterations
    total_iterations = n_epochs * len(train_loader)
    target_iteration = total_iterations - cats if cats is not None else 0

    itr = 0


    for epo in range(n_epochs):
        for data, target in train_loader:
            model.train()
            itr += 1
            data, target = data.to(device), target.to(device)

            outputs = model(data)
            loss = criterion(outputs, target)
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if cats is None or itr > target_iteration:
                with torch.no_grad():
                    model.eval()
                    logits = model(X)                          # (N, 19), on device
                    prob = torch.sigmoid(logits).detach()        # keep as tensor, on device

                    threshold = find_best_thresholds(prob.cpu().numpy(), y.cpu().numpy())  # (19,)
                    threshold_t = torch.as_tensor(threshold, dtype=prob.dtype, device=prob.device)

                    decision = (prob > threshold_t).to(cat_all.dtype)   # stays on device, matches cat_all dtype
                    cat_all += decision

            if itr % 100 == 0:
                print(f"Epoch: {epo} | Iteration: {itr} | Train Loss: {loss.item()}")

        # pass through test data
        with torch.no_grad():
            model.eval()
            test_loss = 0
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                loss = criterion(outputs, target)
                test_loss += loss.item() * data.size(0)
            test_loss /= len(test_loader.dataset)
            loss_history.append(test_loss)
            print(f"Epoch: {epo} | Test Loss: {test_loss}")

        torch.save({'model': model.state_dict()}, f'models/model_epoch_{epo}.pt')

        # if last 3 test lost did not decrease, stop training early
        if len(loss_history) >= 3:
            if loss_history[-1] > loss_history[-2] and loss_history[-1] > loss_history[-3]:
                print(f"Early stopping at epoch {epo} due to no improvement in test loss")
                break
    
    #plot test loss history
    plt.figure(figsize=(5, 2))
    plt.plot(loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('Test Loss')
    plt.title('Test Loss History')
    plt.savefig('plots/test_loss_history.png')

    prob = torch.sigmoid(logits).detach()
                    
    # Calculate your stability based on the rolling 100-step history (or not)
    stab = shanon_stability_multilabel(cat_all, cats)

    # calculate means and variances for mahalanobis
    stats = compute_class_means_and_covariance(model, train_loader, output_dim , device)

    return model, stab, prob, stats, y


# In[ ]:

counts = torch.tensor(class_appear_counts.values, dtype=torch.float32)
pos_weight = torch.log(len(train_imgs) / (counts + 1)).to(device)

print("Start training model")
model, stab, prob, stats, y = train_model(8, train_loader, test_loader, pos_weight, cats=100, output_dim=19)

# In[ ]:


torch.save({'model': model}, 'models/model_resnet18_224x224_8ep.pt')
torch.save({'stability': stab}, 'models/stability_resnet18_224x224_8ep.pt')
torch.save({'prob': prob}, 'models/sigmoid_resnet18_224x224_8ep.pt')
torch.save({'test_labels': test_labels}, 'models/test_labels_resnet18_224x224_8ep.pt')
torch.save({'stats': stats}, 'models/stats_resnet18_224x224_8ep.pt')



# In[ ]:


# # get OOD data for hypersphere_data
# X = get_grid_OOD(input_dim = 2, num_points_per_dim=50) # set num_points_per_dim^2>200 since we need at least 200 pts
# idx = np.random.choice(X.numpy().shape[0], size=200, replace=False)
# x_ood = X.numpy()[idx]

# # get adverserial data from hypersphere_data_noisy 
# # (not sure it makes sense for hypersphere_data ...)
# x, y = next(iter(hypersphere_data_noisy(2, 3, 200, batch_size = 200, seed = 4)))
# atk = torchattacks.FGSM(model, eps=0.05)
# x_adv = atk(x,y)

# # concat data
# x_neg = torch.cat((x_adv, torch.from_numpy(x_ood)), dim=0) 
# x_pos, y = next(iter(hypersphere_data_noisy(2, 3, 400, batch_size = 400, seed = 3)))

# # use these data to train alpha
# alpha = compute_alpha(model, x_pos, x_neg, stats)

# # generate data in input space and get mohalanobis confidence for all points
# X = get_grid(input_dim = 2, num_points_per_dim=150)
# moha = mohalanobis(X, model, stats, alpha)

