import torch
import torch.nn as nn
import torchvision.models as models
import cv2
import numpy as np
 
class DualBackboneHPA(nn.Module):
    def __init__(self, num_classes=19):
        super(DualBackboneHPA, self).__init__()
        
        # 1. ResNet backbone
        self.resnet = models.resnet18(weights=None)
        # Modify first conv to accept 4-channel input (RGBY) if needed
        self.resnet.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Identity()  # Remove standard classification head
        
        # 2. EfficientNet backbone (e.g., EfficientNet-B3)
        self.effnet = models.efficientnet_b3(weights=None)
        # Adjust input channels and classifier head
        self.effnet.features[0][0] = nn.Conv2d(4, 40, kernel_size=3, stride=2, padding=1, bias=False)
        self.effnet.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(1536, num_classes) # Final bias vector length = 19
        )

    def forward(self, x):
        res_feat = self.resnet(x)
        out = self.effnet(x)
        return out

    def load_hpa_sample_4chan(image_id, data_dir):
        red = cv2.imread(f"{data_dir}/{image_id}_red.png", cv2.IMREAD_GRAYSCALE)
        green = cv2.imread(f"{data_dir}/{image_id}_green.png", cv2.IMREAD_GRAYSCALE)
        blue = cv2.imread(f"{data_dir}/{image_id}_blue.png", cv2.IMREAD_GRAYSCALE)
        yellow = cv2.imread(f"{data_dir}/{image_id}_yellow.png", cv2.IMREAD_GRAYSCALE)
        
        # Stack all 4 filters into shape: (4, H, W)
        four_chan_img = np.stack([red, green, blue, yellow], axis=-1)
        return torch.from_numpy(four_chan_img).permute(2, 0, 1).float() / 255.0