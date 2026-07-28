# import os
# import torch
from torchvision import transforms
from PIL import Image
# import kagglehub
import pandas as pd
from functions import *

DEBUG=False

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Download/Load Model Weights via kagglehub
print("Downloading/loading model weights...")
# model_path = kagglehub.model_download("vickykiwisy/res18b3-block-3456-5050/PyTorch/default")
# weight_file = os.path.join(model_path, "best_model_train_RES18_B3_BLOCK3456.pth")

# or using downloaded weights in cluster
model_path = '/home/ykc0662/.cache/kagglehub/models/vickykiwisy/res18b3-block-3456-5050/pyTorch/default/2/best_model_train_RES18_B3_BLOCK3456.pth'
weight_file = torch.load(model_path, map_location=device)

# Initialize model
model = DualBackboneHPA(num_classes=19)

# Load model weights
missing_keys, unexpected_keys = model.load_state_dict(weight_file, strict=False)

if DEBUG:
    print(f"Missing keys length: {len(missing_keys)}")
    print(f"Unexpected keys length: {len(unexpected_keys)}\n")

    print(f"Missing keys: {missing_keys}")
    print(f"Unexpected keys: {unexpected_keys}\n")

    print("Model keys example:", list(model.state_dict().keys())[:5])
    print("Checkpoint keys example:", list(weight_file.keys())[:5])


print("put model to device")
model.to(device)
print("put model to eval mode")
model.eval()


# testing model on a subset of images
print("read subset csv")
subset_df = pd.read_csv('/gpfs/projects/b1042/AmaralLab/shin/DNN-Decision-Uncertainty-Quantification/train_subset.csv')
data_dir = '/gpfs/projects/b1042/AmaralLab/shin/DNN-Decision-Uncertainty-Quantification/data/train_subset_images'
test_img = []

# used to transform images smaller
transform = transforms.Compose([
    transforms.Resize((224, 224)),
])

# Load images
print("Load images...")
test_img = []
with torch.no_grad():
    for image_id in subset_df['ID']:
        img = load_hpa_sample_4to3(image_id, data_dir)  # (3, H, W)
        img = transform(img)  # resize to (3, 224, 224)
        test_img.append(img)

# turn into tensor and put into model
print("Put images tensor in model")
batch = torch.stack(test_img).to(device)  # (N, 3, 224, 224)
outputs = model(batch)


print("printing outputs")

print(outputs.shape)



# # Define Image Preprocessing (Resize to lower resolution, e.g., 224x224 or 512x512)
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),  # Lower resolution to save memory & speed up
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])





# # Process a Subset of Images
# image_dir = "./data/train_subset_images"  # Folder containing your subset of images
# output_results = []

# if os.path.exists(image_dir):
#     image_files = os.listdir(image_dir)[:50]  # Grab only the first 50 images for now!
    
#     with torch.no_grad():
#         for img_name in image_files:
#             img_path = os.path.join(image_dir, img_name)
#             try:
#                 img = Image.open(img_path).convert('RGB')
#                 tensor = transform(img).unsqueeze(0).to(device)
                
#                 output = model(tensor)
#                 # Process predictions...
#                 output_results.append((img_name, output.cpu().numpy()))
#             except Exception as e:
#                 print(f"Error processing {img_name}: {e}")

#     print(f"Successfully processed {len(output_results)} images!")