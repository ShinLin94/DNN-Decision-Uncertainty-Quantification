import torch
from torch import nn
# from torch import optim
# from torch.nn.utils import parameters_to_vector
# from torch.nn.functional import gelu

import numpy as np
# import matplotlib.pyplot as plt
# from tqdm import tqdm

# from copy import deepcopy
# from itertools import permutations

# from helpers import *

# from collections import deque
# from torch.utils.data import TensorDataset, DataLoader

from sklearn.linear_model import LogisticRegression


def shanon_stability(cat_all, output_dim):
    # cat_all.shape = (len(X), output_dim)
    # get % of a point being classified as a class
    cat_all = cat_all / cat_all.sum(dim=1).unsqueeze(1)

    # stability of each point on grid from Shannon formula (len(X), 1)
    # elementwise mult -> cat_all * torch.log(cat_all + 1e-10) 
    stability = 1 + torch.sum(cat_all * torch.log(cat_all + 1e-10) / np.log(output_dim), dim=1)

    return stability.detach().cpu().numpy()

def get_features(model, x, layer_types=(nn.GELU,), detach=True): 
    """ 
    Change nn.Linear to switch feature layer to extract (features before activation fn)
    Or if use other layer types (RELU,GELU,...etc)
    """
    features = {}
    hooks = []

    def make_hook(name):
        def hook(module, input, output):
            features[name] = output.detach() if detach else output
        return hook

    for i, layer in enumerate(model.net):
        if isinstance(layer, layer_types):
            hooks.append(layer.register_forward_hook(make_hook(i)))

    if detach:
        with torch.no_grad():
            model(x)
    else:
        model(x)

    for h in hooks:
        h.remove()

    if layer_types==(nn.Linear,):
        keys = sorted(features.keys())
        selected_keys = keys[1:-1]  # drop first and last layer
        features = {k: features[k] for k in selected_keys}

    return features

def compute_class_means_and_covariance(model, train_loader, num_classes, device):
    model.eval()
    feats_per_layer = {}
    labels_all = []

    for x, y in train_loader:
        x = x.to(device)
        feats = get_features(model, x)  # {layer_idx: [batch, d]}
        for l, f in feats.items():
            feats_per_layer.setdefault(l, []).append(f.cpu())
        labels_all.append(y)

    labels_all = torch.cat(labels_all)
    stats = {}
    for l, flist in feats_per_layer.items():
        f = torch.cat(flist)  # [N, d_l]
        mu = torch.stack([f[labels_all == c].mean(dim=0) for c in range(num_classes)])
        centered = torch.cat([f[labels_all == c] - mu[c] for c in range(num_classes)])
        sigma = (centered.T @ centered) / f.shape[0]
        stats[l] = {'mu': mu, 'sigma_inv': np.linalg.pinv(sigma)}

    return stats

def compute_alpha(model, X_pos, X_neg, stats, device):
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
                mu = stats[l]['mu'].to(device)
                sigma_inv = stats[l]['sigma_inv'].to(device)
                diff = f.unsqueeze(1) - mu.unsqueeze(0)
                dist = torch.einsum('bcd,de,bce->bc', diff, sigma_inv, diff)
                scores.append(-dist.min(dim=1).values)
            return torch.stack(scores, dim=1).cpu()  # [batch, L]

    scores_in = layer_scores(X_pos)
    scores_ood = layer_scores(X_neg)

    X_lr = torch.cat([scores_in, scores_ood]).numpy()
    y_lr = torch.cat([torch.ones(len(scores_in)), torch.zeros(len(scores_ood))]).numpy()

    clf = LogisticRegression()
    clf.fit(X_lr, y_lr)

    return [clf.coef_.flatten(), clf.intercept_.item()]


def mohalanobis(X, model, stats, alpha, device, eps=1e-5):
    model.eval()
    # alpha = torch.as_tensor(alpha, dtype=torch.float32)

    # get features of data at every hidden layer
    feats = get_features(model, X)  # {layer_idx: [x, d]}

    # store confidence at every layer
    M_l = torch.empty(len(X), len(feats))

    # for each layer, find confidence
    for i, (l, f) in enumerate(feats.items()):
        mu = stats[l]['mu'].to(device) # [C, d_l]
        sigma_inv = stats[l]['sigma_inv'].to(device) # [d_l, d_l]

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
        
    final = M_l @ alpha[0] + alpha[1]   # [x]
    final_prob = torch.sigmoid(final) 

    return final_prob    
            