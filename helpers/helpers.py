# This file is written by Cameron Mackenzie
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.nn.functional import nll_loss

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from copy import deepcopy

def hypersphere_data(dim_in, dim_out, N_samples, batch_size, seed):
    """
    Generates N-dimensional hypersphere problem data.

    Unit vector directions are drawn from standard normal in dim_in dimensions.
    Then vectors are scaled uniformly by radius 0 < r < 1.
    Radii are assigned to dim_out even classes (e.g. dim_out = 2 gives classes (0,0.5), (0.5,1.0))

    Function returns DataLoader class containing generated samples.
    """
    torch.manual_seed(seed)

    u = torch.randn(N_samples, dim_in)
    u = u / torch.linalg.vector_norm(u, dim=1).unsqueeze(1)
    
    r = torch.rand(N_samples)
    
    data = u * r.unsqueeze(1) # Scale unit directions by radii
    labels = torch.bucketize(r, torch.linspace(0, 1, dim_out+1)) - 1 # Evenly divide radii into classes
    
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(TensorDataset(data, labels), 
                      batch_size=batch_size, shuffle=True, generator=g)

class MLP(nn.Module):
    def __init__(self, layers, activation = nn.ReLU, bias=True):
        """
        layers: Iterable with first item being input dim 
        and last being output dim. Intermediate dims optional.
        For example: (28*28, 256, 10)
        """
        super().__init__()
        self.net = nn.Sequential(nn.Flatten())

        for dim_in, dim_out in zip(layers[:-2], layers[1:-1]):
            self.net.append(nn.Linear(dim_in, dim_out, bias=bias))
            self.net.append(activation())

        self.net.append(nn.Linear(layers[-2], layers[-1], bias=bias))

    def forward(self, x):
        return self.net(x)

def fgsm(img, eps, gradient):
    """
    Generates an adversarial image by FGSM for given
    epsilon, data gradient.
    """

    signs = gradient.sign()
    new_img = img + eps * signs
    new_img = torch.clamp(new_img, 0, 1)

    return new_img

def fgsm_dataset(model, device, loader, img_shape, eps, N):
    """
    Returns images manipulated by FGSM, along with original labels.

    loader: must have batch_size of 1
    img_shape: tuple containing size of single image in loader, e.g. (3, 32, 32)
    eps: FGSM parameter. Higher value => greater perturbation.
    N: (max) number of samples to return. Only loops through loader once,
        so it is sometimes possible for <N samples to be returned.
    """
    model.to(device)
    adv_images = torch.zeros(0, *img_shape).to(device)
    adv_labels = torch.zeros(0).to(device)

    for i, (image, label) in tqdm(enumerate(loader)):
        if len(adv_images) == N:
            break

        image, label = image.to(device), label.to(device)
        image.requires_grad = True

        output = model(image)

        # Only include samples that are initially predicted correctly
        prediction = output.argmax(dim=1)
        if prediction.item() != label.item():
            continue

        loss = nll_loss(output, label)
        model.zero_grad()
        loss.backward()

        gradient = image.grad.detach()

        # When model is very confident in prediction, FGSM doesn't generate any noise
        # Skip these examples
        if gradient.abs().sum().item() == 0:
            continue
        
        new_image = fgsm(image, eps, gradient)

        adv_images = torch.cat((adv_images, new_image.detach()))
        adv_labels = torch.cat((adv_labels, label))
    
    return adv_images, adv_labels
    