import torch
import torch.nn as nn
import torchvision.models as models
import cv2
import numpy as np
 
class DualBackboneHPA(nn.Module):
    def __init__(self, num_classes=19):
        super(DualBackboneHPA, self).__init__()

        # # If using 4 channels (requires training)
        # # ResNet backbone
        # self.resnet = models.resnet18(weights=None)
        # # Modify first conv to accept 4-channel input (RGBY)
        # self.resnet.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # self.resnet.fc = nn.Identity()  # output: 512-dim features

        
        # # EfficientNet backbone (e.g., EfficientNet-B3)
        # self.effnet = models.efficientnet_b3(weights=None)
        # # Adjust input channels and classifier head
        # self.effnet.features[0][0] = nn.Conv2d(4, 40, kernel_size=3, stride=2, padding=1, bias=False)
        # self.effnet.classifier = nn.Identity()  # output: 1536-dim features

        # self.head = nn.Sequential(
        #     nn.Dropout(p=0.3),
        #     nn.Linear(512 + 1536, num_classes)
        # )

        # standard 3 channels (using pretrained weights)
        # ResNet backbone with 19-class head
        self.resnet = models.resnet18(weights=None)
        self.resnet.fc = nn.Linear(512, num_classes)  # Shape matches [19, 512]
        
        # EfficientNet backbone with 19-class head
        self.effnet = models.efficientnet_b3(weights=None)
        self.effnet.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(1536, num_classes)              # Shape matches [19, 1536]
        )
    
    def forward(self, x):
        # Obtain predictions from both backbones (each outputs shape: [batch_size, 19])
        res_probs = torch.sigmoid(self.resnet(x))
        eff_probs = torch.sigmoid(self.effnet(x))

        # Avg predictions
        return (res_probs + eff_probs) / 2.0  # returns probabilities, not logits (get rid of sigmoid to get logits)

def load_hpa_sample_4to3(image_id, data_dir):
    """Combines 4 HPA color filters into a 3-channel RGB image normalized to [0, 1]."""
    # Load 4 monochrome grayscale images (0-255)
    channels = {}
    for name in ['red', 'green', 'blue', 'yellow']:
        path = f"{data_dir}/{image_id}_{name}.png"
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {path} (image_id={image_id})")
        channels[name] = img.astype(np.float32)

    r_filter, g_filter, b_filter, y_filter = channels['red'], channels['green'], channels['blue'], channels['yellow']


    # Blend yellow into red and green channels
    red_channel = r_filter + (y_filter * 0.5)
    green_channel = g_filter + (y_filter * 0.5)
    blue_channel = b_filter

    # Normalize max brightness to 255.0 if clipped
    max_val = max(red_channel.max(), green_channel.max(), blue_channel.max())
    if max_val > 255.0:
        red_channel = (red_channel * 255.0) / max_val
        green_channel = (green_channel * 255.0) / max_val
        blue_channel = (blue_channel * 255.0) / max_val

    # Stack into 3-channel RGB array: shape (H, W, 3)
    rgb_image = np.stack([red_channel, green_channel, blue_channel], axis=-1)

    # Convert to PyTorch float tensor (3, H, W) normalized to [0, 1]
    tensor_img = torch.from_numpy(rgb_image).permute(2, 0, 1).float() / 255.0
    return tensor_img


def parse_labels_to_multihot(label_str, num_classes=19):
    """Converts '16|13' into a 19-dim multi-hot vector."""
    indices = [int(x) for x in label_str.split('|')]
    vec = np.zeros(num_classes, dtype=np.float32)
    vec[indices] = 1.0
    return vec