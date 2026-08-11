
import gc

import torch
from torch import nn
# from torch import optim
# from torch.nn.utils import parameters_to_vector
# from torch.nn.functional import gelu, relu

from PIL import Image
import numpy as np
# import matplotlib.pyplot as plt
# from tqdm import tqdm

# from copy import deepcopy
# from itertools import permutations

# from helpers import *

# from collections import deque
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torchvision
import torchvision.transforms as T

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from scipy.stats import chi2

IMG_SIZE = 224  
NORM_MEAN = (0.077, 0.046, 0.075)
NORM_STD = (0.135, 0.097, 0.169)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def compute_dataset_stats(images_uint8_or_float, already_scaled_0_1=False, chunk_size=6000):
    """
    Computes per-channel mean/std over your training set, in chunks, so it
    never materializes the full tensor as float32 at once (a 156k x 3 x 224 x 224
    uint8 tensor is ~24GB; converting the whole thing to float32 is ~94GB and
    will OOM on most nodes).
 
    images_uint8_or_float: tensor of shape (N, 3, H, W), either uint8 [0,255]
        or float already in [0,1] (set already_scaled_0_1=True in that case).
    """
    n = images_uint8_or_float.shape[0]
    channels = images_uint8_or_float.shape[1]
 
    # Welford-style running sums, computed per chunk to bound memory
    sum_ = torch.zeros(channels, dtype=torch.float64)
    sum_sq = torch.zeros(channels, dtype=torch.float64)
    count = 0
 
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = images_uint8_or_float[start:end].float()
        if not already_scaled_0_1:
            chunk = chunk / 255.0
 
        # sum over batch, height, width -> per-channel totals
        pixels_per_channel = chunk.shape[0] * chunk.shape[2] * chunk.shape[3]
        sum_ += chunk.sum(dim=(0, 2, 3)).double()
        sum_sq += (chunk ** 2).sum(dim=(0, 2, 3)).double()
        count += pixels_per_channel
 
        del chunk  # free this chunk's float memory before the next iteration
 
    mean = sum_ / count
    # var = E[x^2] - E[x]^2
    var = (sum_sq / count) - (mean ** 2)
    std = torch.sqrt(var)
 
    return tuple(mean.float().tolist()), tuple(std.float().tolist())


def find_best_thresholds(probs, labels, thresholds=np.arange(0.05, 0.95, 0.05)):
    """probs, labels: (N, num_classes) numpy arrays"""
    best_thresholds = np.zeros(probs.shape[1])
    for c in range(probs.shape[1]):
        best_f1, best_t = 0, 0.5
        for t in thresholds:
            preds_c = (probs[:, c] > t).astype(int)
            f1 = f1_score(labels[:, c], preds_c, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        best_thresholds[c] = best_t

    # returns best thresholds for each class (output_dim,1)
    return best_thresholds


def shannon_stability_multilabel(cat_all, cats):   
    # cat_all.shape = (len(X), output_dim)
    # get % of a class being classified as positive for a point
    cat_all = cat_all / cats

    # stability of each class (binary prob) for each point on grid from Shannon formula (len(X), output_dim)
    # elementwise mult -> cat_all * torch.log(cat_all + 1e-10) 
    stabilities = 1 + (cat_all * torch.log(cat_all + 1e-10) + (1 - cat_all) * torch.log(1 - cat_all + 1e-10)) / np.log(2)
    stability = stabilities.mean(dim=1)

    return stability.detach().cpu().numpy()


def get_features(model, x, layers=("layer1", "layer2", "layer3", "layer4"),
                         detach=True):
    """ Made for resnet:
    Hooks the output of each named residual stage in `layers` and returns
    spatially-pooled feature vectors: {layer_name: [batch, channels]}.
 
    model: a torchvision.models.resnet* instance (or anything with attributes
        matching the names in `layers`, e.g. model.layer1, model.layer2, ...)
    x: input batch, already on the same device as model
    layers: names of submodules to hook (default: the 4 standard ResNet stages)
    detach: if True, runs forward pass under torch.no_grad() and detaches features
        (use detach=False only if you need gradients through these features,
        e.g. for adversarial generation against the Mahalanobis score itself)
    """
    features = {}
    hooks = []
 
    def make_hook(name):
        def hook(module, input, output):
            # output: [B, C, H, W] -> global average pool -> [B, C]
            pooled = output.mean(dim=(2, 3))
            features[name] = pooled.detach() if detach else pooled
        return hook
 
    for name in layers:
        layer = getattr(model, name)
        hooks.append(layer.register_forward_hook(make_hook(name)))
 
    if detach:
        with torch.no_grad():
            model(x)
    else:
        model(x)
 
    for h in hooks:
        h.remove()
 
    return features
 
 
def compute_class_means_and_covariance(model, train_loader, num_classes,
                                                device, layers=("layer1", "layer2",
                                                                 "layer3", "layer4")):
    """
    Same role as original compute_class_means_and_covariance from 03...py, but:
      1. uses get_features_resnet (spatially-pooled conv features) instead of
         hooking activation layers on a flat nn.Sequential
      2. handles MULTI-LABEL targets
 
    For multi-label data, "class-conditional" mean/covariance is ambiguous when
    a sample belongs to multiple classes at once. The standard adaptation (as
    used in multi-label OOD work) is: for each class c, gather all samples where
    y[:, c] == 1, and treat that as class c's population -- a sample can
    contribute to multiple classes' statistics.
    """
    model.eval()
    feats_per_layer = {l: [] for l in layers}
    labels_all = []
 
    for x, y in train_loader:
        x = x.to(device)
        feats = get_features(model, x, layers=layers)
        for l, f in feats.items():
            feats_per_layer[l].append(f.cpu())
        labels_all.append(y.cpu())  # y: [batch, num_classes] multi-hot
 
    labels_all = torch.cat(labels_all)  # [N, num_classes]
 
    stats = {}
    for l in layers:
        f = torch.cat(feats_per_layer[l])  # [N, d_l]
        d = f.shape[1]
 
        mu = torch.zeros(num_classes, d)
        # accumulate pooled within-class scatter across all classes
        scatter_sum = torch.zeros(d, d)
        total_n = 0
 
        for c in range(num_classes):
            mask = labels_all[:, c] == 1
            n_c = mask.sum().item()
            if n_c == 0:
                # no samples for this class in train_loader -- leave mu[c] as zeros
                # and skip its contribution to the pooled covariance
                continue
            f_c = f[mask]
            mu[c] = f_c.mean(dim=0)
            centered = f_c - mu[c]
            scatter_sum += centered.T @ centered
            total_n += n_c
 
        sigma = scatter_sum / total_n  # pooled within-class covariance, as in Lee et al.
        stats[l] = {
            'mu': mu,
            'sigma_inv': torch.linalg.pinv(sigma)
        }
 
    return stats







# def compute_class_means_and_covariance(model, train_loader, num_classes, device=DEVICE):
#     model.eval()
#     feats_per_layer = {}
#     labels_all = []

#     for x, y in train_loader:
#         x = x.to(device)
#         feats = get_features(model, x)  # {layer_idx: [batch, d]}
#         for l, f in feats.items():
#             feats_per_layer.setdefault(l, []).append(f.cpu())
#         labels_all.append(y)

#     labels_all = torch.cat(labels_all)
#     stats = {}
#     for l, flist in feats_per_layer.items():
#         f = torch.cat(flist)  # [N, d_l]
#         mu = torch.stack([f[labels_all == c].mean(dim=0) for c in range(num_classes)])
#         centered = torch.cat([f[labels_all == c] - mu[c] for c in range(num_classes)])
#         sigma = (centered.T @ centered) / f.shape[0]
#         stats[l] = {'mu': mu, 'sigma_inv': torch.linalg.pinv(sigma)}

#     return stats

def compute_alpha(model, X_pos, X_neg, stats, device=DEVICE):
    """
    X_indist: [N1, input_dim] in-distribution validation samples
    X_ood: [N2, input_dim] OOD or adversarial validation samples
    Returns: alpha, array of length L (layer weights), in sorted(stats.keys()) order
    """
    layers = sorted(stats.keys())

    def layer_scores(X):
        model.eval()
        with torch.no_grad():
            feats = get_features(model, X.to(device))
            scores = []
            for l in layers:
                f = feats[l]
                mu = stats[l]['mu'] # [C, d]   make mu and sigma_inv torch tensors if not already
                mu = torch.as_tensor(mu, dtype=torch.float32)
                sigma_inv = stats[l]['sigma_inv'] # [d, d]
                sigma_inv = torch.as_tensor(sigma_inv, dtype=torch.float32)
                diff = f.unsqueeze(1) - mu.unsqueeze(0) # bcd - cd
                dist = torch.einsum('bcd,de,bce->bc', diff, sigma_inv, diff)
                scores.append(-dist.min(dim=1).values)
            return torch.stack(scores, dim=1).cpu()  # [batch, L]

    scores_in = layer_scores(X_pos)
    del X_pos 
    gc.collect()
    scores_ood = layer_scores(X_neg)
    del X_neg
    gc.collect()

    X_lr = torch.cat([scores_in, scores_ood]).numpy()
    df = [stats[l]['mu'].to(device).shape[1] for l in layers]
    X_lr = chi2.cdf(X_lr, df=df)
    y_lr = torch.cat([torch.ones(len(scores_in)), torch.zeros(len(scores_ood))]).numpy()

    clf = LogisticRegression()
    clf.fit(X_lr, y_lr)

    return [clf.coef_.flatten(), clf.intercept_.item()]


def mahalanobis(X, model, stats, alpha, device=DEVICE, eps=0):
    model.eval()
    # alpha = torch.as_tensor(alpha, dtype=torch.float32)

    # get features of data at every hidden layer
    feats = get_features(model, X)  # {layer_idx: [x, d]}

    # store confidence at every layer
    M_l = torch.empty(len(X), len(feats))

    # for each layer, find confidence
    for i, (l, f) in enumerate(feats.items()):
        mu = stats[l]['mu']
        mu = torch.as_tensor(mu, dtype=torch.float32).to(device) # [C, d_l]
        sigma_inv = stats[l]['sigma_inv']
        sigma_inv = torch.as_tensor(sigma_inv, dtype=torch.float32).to(device) # [d_l, d_l]
        
        # find closest class (max)
        with torch.no_grad():
            diff = f.unsqueeze(1) - mu.unsqueeze(0) # [x, C, d_l] 
            # for each data (x), do 1.d@d.e@e.1 (guass exp M(x)) for all c (classes)
            dist = torch.einsum('xcd,de,xce->xc', diff, sigma_inv, diff) 
            closest_c = dist.argmin(dim=1) # [x]
        
        # perturb input using gradient of score at closest class
        if eps > 0:
            X_in = X.clone().requires_grad_(True)
            feats_g = get_features(model, X_in, detach=False) # with grad to calc grad
            f_g = feats_g[l]
            mu_c = mu[closest_c] # [x, d_l]
            diff_c = f_g - mu_c
            score_closest = torch.einsum('xd,de,xe->x', diff_c, sigma_inv, diff_c) # [x]
            score_closest.sum().backward() # sum so output is scalar (still x diff input variables)
            X_pert = (X_in - eps * X_in.grad.sign()).detach()
        else:
            X_pert = X

        # find the uncertainty using mahalanobis distance of new closest class after perterbation
        with torch.no_grad():
            feats_p = get_features(model, X_pert)
            f_p = feats_p[l]
            diff_p = f_p.unsqueeze(1) - mu.unsqueeze(0)
            dist_p = torch.einsum('xcd,de,xce->xc', diff_p, sigma_inv, diff_p) #[x, C]
            M_l[:,i] = -dist_p.min(dim=1).values # [x, L]

    df = [stats[l]['mu'].to(device).shape[1] for l in sorted(stats.keys())]
    confs = chi2.cdf(M_l, df=df)
    conf = confs @ alpha[0] + alpha[1]   # [x]

    return conf    # numpy


def get_cifar10_ood_loader(root="./data", batch_size=64, train=False,
                            num_workers=2, download=True):
    """
    Loads CIFAR-10 resized/normalized to match the in-distribution HPA pipeline.
    Returns a DataLoader yielding (image, label) — label is CIFAR's class id,
    which we will ignore since these are only used as OOD negatives.
    """
    transform = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ])

    dataset = torchvision.datasets.CIFAR10(
        root=root, train=train, download=download, transform=transform
    )

    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                       num_workers=num_workers, pin_memory=True)


class NoiseDataset(Dataset):
    """
    Generates random noise images on the fly. `mode` controls the noise
    distribution:
      - "gaussian": N(0, 1) then normalized like real data
      - "uniform":  U(0, 1) then normalized like real data
    """
    def __init__(self, n_samples=2000, img_size=IMG_SIZE, channels=3,
                 mode="gaussian", mean=NORM_MEAN, std=NORM_STD, seed=None):
        self.n_samples = n_samples
        self.img_size = img_size
        self.channels = channels
        self.mode = mode
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        shape = (self.channels, self.img_size, self.img_size)
        if self.mode == "gaussian":
            # centered at 0.5 like real [0,1] pixel intensities,
            # std=0.25 keeps ~95% of values inside [0,1]; clamp handles the tails
            img = torch.randn(shape, generator=self.generator) * 0.25 + 0.5 
            img = torch.clamp(img, 0, 1)
            img = (img - self.mean) / self.std 
        elif self.mode == "uniform":
            img = torch.rand(shape, generator=self.generator)  # [0, 1]
            img = (img - self.mean) / self.std
        else:
            raise ValueError(f"Unknown noise mode: {self.mode}")
        label = -1  # sentinel label, these have no real class
        return img, label


def get_noise_ood_loader(n_samples=2000, batch_size=64, img_size=IMG_SIZE,
                          channels=3, mode="gaussian", num_workers=2, seed=None):
    dataset = NoiseDataset(n_samples=n_samples, img_size=img_size,
                            channels=channels, mode=mode, seed=seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                       num_workers=num_workers, pin_memory=True)


def fgsm_attack(model, images, labels, epsilon, criterion=nn.BCEWithLogitsLoss(), device=DEVICE):
    """
    Single-step FGSM: x_adv = x + epsilon * sign(grad_x L(model(x), y))
    `images` should already be normalized the same way as your training data.
    `labels` should be the ground-truth (or predicted, if unlabeled) targets —
    for multi-label HPA data, use BCEWithLogitsLoss and multi-hot labels.
    """

    images = images.clone().detach().to(device).requires_grad_(True)
    labels = labels.to(device)

    model.eval()
    outputs = model(images)
    loss = criterion(outputs, labels.float())

    model.zero_grad()
    loss.backward()

    with torch.no_grad():
        perturbation = epsilon * images.grad.sign()
        adv_images = images + perturbation
        # NOTE: no clamp to [0,1] here since inputs are already normalized;
        # if you need pixel-space clamping, unnormalize -> clamp -> renormalize.

    return adv_images.detach().cpu()


def get_adversarial_loader(model, in_dist_loader, epsilon=0.02, device=DEVICE, criterion=nn.BCEWithLogitsLoss(), max_batches=None):
    """
    Runs FGSM over an existing in-distribution DataLoader and returns a new
    DataLoader of adversarial examples (same batch_size as input loader).
    This is your alpha_adv negative set for the Mahalanobis detector.
    """
    model = model.to(device)
    adv_images_list, labels_list = [], []

    for i, (images, labels) in enumerate(in_dist_loader):
        if max_batches is not None and i >= max_batches:
            break
        adv_batch = fgsm_attack(model, images, labels, epsilon,
                                 criterion=criterion, device=device)
        adv_images_list.append(adv_batch)
        labels_list.append(labels)

    adv_images_tensor = torch.cat(adv_images_list, dim=0)
    labels_tensor = torch.cat(labels_list, dim=0)

    adv_dataset = TensorDataset(adv_images_tensor, labels_tensor)
    return DataLoader(adv_dataset, batch_size=in_dist_loader.batch_size,
                       shuffle=False, pin_memory=True)
