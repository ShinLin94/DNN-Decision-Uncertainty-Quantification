import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.nn.functional import nll_loss

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from copy import deepcopy

def ordered_loader(dataset, batch_size, seed):
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=g)

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

def checkerboard_data(dim_in, dim_out, N_samples, batch_size, seed):
    torch.manual_seed(seed)

    data = torch.rand(N_samples, dim_in)
    target = torch.floor(data * dim_out).to(int).sum(dim=1) % dim_out
    dataset = TensorDataset(data, target)

    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=g)

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

@torch.no_grad()
def plot_three_models(m1, m2, m3, 
                      test_data, test_labels,
                      ax,
                      res=21, colors=["C0", "C0", "C0"],
                      loss_bar = False,
                      fig = None):
    """Provide fig if loss_bar = True."""
    
    w1 = parameters_to_vector(m1.parameters()).detach()
    w2 = parameters_to_vector(m2.parameters()).detach()
    w3 = parameters_to_vector(m3.parameters()).detach()
    
    # Define the plane - w1 at (0,0) and w2 along x-axis
    v0 = w2 - w1
    v0 = v0 / torch.norm(v0)
    
    v1 = w3 - w1 # Temp
    v1 = v1 - torch.dot(v0, v1) * v0
    v1 = v1 / torch.norm(v1)
    
    plane = lambda s,t : w1 + v0 * s + v1 * t
        
    # Find where w2, w3 lie wrt plane parameterization
    w2_pt = torch.linalg.lstsq(
            torch.cat((v0.unsqueeze(1), v1.unsqueeze(1)), dim=1),
            w2 - w1
        ).solution.numpy()
    w3_pt = torch.linalg.lstsq(
            torch.cat((v0.unsqueeze(1), v1.unsqueeze(1)), dim=1),
            w3 - w1
        ).solution.numpy()

    # Plot losses
    padding = 0.1
    
    width = max(0, w2_pt[0], w3_pt[0]) - min(0, w2_pt[0], w3_pt[0])
    height = max(0, w2_pt[1], w3_pt[1]) - min(0, w2_pt[1], w3_pt[1])
    
    # Define size and center point of loss image
    im_half_width = width * (1 + padding) / 2
    im_half_height = height * (1 + padding) / 2
    
    im_x_c = width / 2
    im_y_c = height / 2
    
    # Make sure image isn't too narrow
    im_half_width = max(im_half_width, 0.2*im_half_height)
    im_half_height = max(im_half_height, 0.2*im_half_width)
    
    im_x = np.linspace(im_x_c - im_half_width, im_x_c + im_half_width, res)
    im_y = np.linspace(im_y_c - im_half_height, im_y_c + im_half_height, res)
    
    # Find losses
    model = deepcopy(m1)
    losses = np.zeros((res,res))
    
    criterion = nn.CrossEntropyLoss()
    
    for i in range(res):
        for j in range(res):
            test_weights = plane(im_x[i], im_y[j])

            vector_to_parameters(test_weights, model.parameters())

            model.train()
            outputs = model(test_data)
            loss = criterion(outputs, test_labels)

            losses[i,j] = loss.item()
    
    # [xmin, xmax, ymin, ymax]
    extent = [im_x[0] - im_half_width / (res-1),
             im_x[-1] + im_half_width / (res-1),
             im_y[0] - im_half_height / (res-1),
             im_y[-1] + im_half_height / (res-1)]

    im = ax.imshow(losses.T, cmap="Spectral", origin="lower", extent=extent, aspect="auto")
    ax.scatter([0, w2_pt[0], w3_pt[0]], [0, w2_pt[1], w3_pt[1]], color=colors, s=20)

    if loss_bar and fig:
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Loss")
    