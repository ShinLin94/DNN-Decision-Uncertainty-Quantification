
import gc

import numpy as np
from pyparsing import line
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import confidence_functions as cf

BASE_DIR = '/gpfs/projects/b1042/AmaralLab/shin/DNN-Decision-Uncertainty-Quantification/models'
CACHE_PATH = '/gpfs/projects/b1042/AmaralLab/shin/DNN-Decision-Uncertainty-Quantification/cached_data_224.pt'
MODEL_TAG = 'resnet18_224x224_8ep'
NUM_CLASSES = 19
CIFAR_ROOT = '/gpfs/projects/b1042/AmaralLab/shin/DNN-Decision-Uncertainty-Quantification/data'
WORKERS = 4  # for DataLoader num_workers

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
# DataLoader generator
g = torch.Generator()
g.manual_seed(SEED)

model = torch.load(f'{BASE_DIR}/model_{MODEL_TAG}.pt', map_location='cpu', weights_only=False)['model']
model.to(device)
model.eval()

stats_maha = torch.load(f'{BASE_DIR}/stats_{MODEL_TAG}.pt', map_location='cpu', weights_only=False)['stats']

cache = torch.load(CACHE_PATH, map_location='cpu', weights_only=False)
all_imgs, all_labels = cache['images'], cache['labels']

idx = torch.load(f'{BASE_DIR}/train_test_indices.pt', map_location='cpu', weights_only=False)
train_indices, test_indices = idx['train_indices'], idx['test_indices']

train_imgs_all = all_imgs[train_indices]
train_labels_all = all_labels[train_indices]
del all_imgs, all_labels, idx, cache
gc.collect()

print("Computing normalization stats from training set (chunked)...")
NORM_MEAN, NORM_STD = cf.compute_dataset_stats(train_imgs_all)
print(f"mean={NORM_MEAN}, std={NORM_STD}")

cf.NORM_MEAN = NORM_MEAN
cf.NORM_STD = NORM_STD
cf.IMG_SIZE = 224

def normalize_uint8_batch(imgs_uint8, mean=NORM_MEAN, std=NORM_STD):
    """imgs_uint8: (N, 3, H, W) uint8 tensor -> normalized float tensor."""
    mean_t = torch.tensor(mean).view(1, -1, 1, 1)
    std_t = torch.tensor(std).view(1, -1, 1, 1)
    imgs = imgs_uint8.float() / 255.0
    return (imgs - mean_t) / std_t


print("Sampling 8000 positive (in-distribution training) examples...")
pos_sample_idx = torch.randperm(len(train_imgs_all), generator=g)[:8000]
pos_imgs = normalize_uint8_batch(train_imgs_all[pos_sample_idx])


print("Generating 8000 FGSM adversarial examples from training data...")
adv_source_idx = torch.randperm(len(train_imgs_all), generator=g)[:4000]
adv_source_imgs = normalize_uint8_batch(train_imgs_all[adv_source_idx])
adv_source_labels = train_labels_all[adv_source_idx]

adv_loader = DataLoader(TensorDataset(adv_source_imgs, adv_source_labels),
                         batch_size=500, shuffle=False, num_workers=WORKERS)
adv_imgs_list = []
for imgs, labels in adv_loader:
    adv_batch = cf.fgsm_attack(model, imgs, labels, epsilon=0.02, device=device)
    adv_imgs_list.append(adv_batch)
adv_imgs = torch.cat(adv_imgs_list, dim=0)[:4000]
print(f"Adversarial negatives: {adv_imgs.shape[0]}")

print(f"Total adv: {adv_imgs.shape[0]} (4000 OOD + 4000 adversarial)")
print(f"Total positives: {pos_imgs.shape[0]}")
del train_imgs_all, train_labels_all, adv_source_imgs, adv_source_labels, adv_imgs_list
gc.collect()  # free up memory before next part

print("Fitting alpha via compute_alpha...")
alpha_adv = cf.compute_alpha(model, pos_imgs, adv_imgs, stats_maha, device=device)
torch.save({'alpha': alpha_adv}, f'{BASE_DIR}/alpha_adv_sig_{MODEL_TAG}.pt')
print(f"Saved trained alpha: {alpha_adv}")