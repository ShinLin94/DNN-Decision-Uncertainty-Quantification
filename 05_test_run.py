import os
import torch
from torchvision import transforms
from PIL import Image
import kagglehub
import pandas as pd
from functions import *


# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Download/Load Model Weights via kagglehub
print("Downloading/loading model weights...")
# model_path = kagglehub.model_download("vickykiwisy/res18b3-block-3456-5050/PyTorch/default")
model_path = '/home/ykc0662/.cache/kagglehub/models/vickykiwisy/res18b3-block-3456-5050/pyTorch/default/2/best_model_train_RES18_B3_BLOCK3456.pth'
# weight_file = os.path.join(model_path, "best_model_train_RES18_B3_BLOCK3456.pth")
weight_file = torch.load(model_path, map_location=device)

# Initialize model
model = DualBackboneHPA(num_classes=19)

# Handle state_dict key wrapper if present
if 'model_state_dict' in weight_file:
    state_dict = weight_file['model_state_dict']
elif 'state_dict' in weight_file:
    state_dict = weight_file['state_dict']
else:
    state_dict = weight_file

# Load model weights
missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

print(f"Missing keys: {len(missing_keys)}")
print(f"Unexpected keys: {len(unexpected_keys)}")


model.to(device)
model.eval()


print(model)


# Define Image Preprocessing (Resize to lower resolution, e.g., 224x224 or 512x512)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Lower resolution to save memory & speed up
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Process a Subset of Images
image_dir = "./data_subset"  # Folder containing your subset of images
output_results = []

if os.path.exists(image_dir):
    image_files = os.listdir(image_dir)[:50]  # Grab only the first 50 images for now!
    
    with torch.no_grad():
        for img_name in image_files:
            img_path = os.path.join(image_dir, img_name)
            try:
                img = Image.open(img_path).convert('RGB')
                tensor = transform(img).unsqueeze(0).to(device)
                
                output = model(tensor)
                # Process predictions...
                output_results.append((img_name, output.cpu().numpy()))
            except Exception as e:
                print(f"Error processing {img_name}: {e}")

    print(f"Successfully processed {len(output_results)} images!")