# import os
# import torch
from torchvision import transforms
from PIL import Image
# import kagglehub
import pandas as pd
from functions import *

from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score
import matplotlib.pyplot as plt


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


print("printing output shape")
print(outputs)
print(outputs.shape)


# get true labels for each sample
gt_labels = np.stack([
    parse_labels_to_multihot(label_str) 
    for label_str in subset_df['Label']
])

# get labels from logits from model
with torch.no_grad():
    preds = (outputs > 0.5).float()  # binary multi-label predictions

    preds_np = preds.cpu().numpy()
    probs_np = outputs.cpu().numpy()

# Macro F1: average F1 per class, unweighted (good with class imbalance)
f1_macro = f1_score(gt_labels, preds_np, average='macro', zero_division=0)

# Micro F1: aggregate across all classes/samples (weights common classes more)
f1_micro = f1_score(gt_labels, preds_np, average='micro', zero_division=0)

# Per-class breakdown, useful for spotting which of the 19 classes are struggling
f1_per_class = f1_score(gt_labels, preds_np, average=None, zero_division=0)

# mAP - uses probabilities, not thresholded predictions, often the headline HPA metric
map_score = average_precision_score(gt_labels, probs_np, average='macro')

print(f"Macro F1: {f1_macro:.3f}")
print(f"Micro F1: {f1_micro:.3f}")
print(f"Macro mAP: {map_score:.3f}")
print("Per-class F1:", np.round(f1_per_class, 3))


# print(gt_labels[:5]) 
# print(preds_np[:5])


class_counts = gt_labels.sum(axis=0)  # sum down the sample axis -> (19,)
sorted_idx = np.argsort(-class_counts)

class_counts_pred = preds_np.sum(axis=0)  # sum down the sample axis -> (19,)
sorted_idx_pred = np.argsort(-class_counts_pred)

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
axs[0].bar(range(len(class_counts)), class_counts[sorted_idx])
axs[0].set_xticks(range(len(class_counts)))
axs[0].set_xticklabels(sorted_idx)
axs[0].set_xlabel('Class Index (sorted by frequency)')
axs[0].set_ylabel('Count')
axs[0].set_title('Class Frequency in Subset (sorted)')

axs[1].bar(range(len(class_counts_pred)), class_counts_pred[sorted_idx_pred])
axs[1].set_xticks(range(len(class_counts_pred)))
axs[1].set_xticklabels(sorted_idx_pred)
axs[1].set_xlabel('Class Index (sorted by frequency)')
axs[1].set_ylabel('Count')
axs[1].set_title('Class Frequency in Prediction (sorted)')

plt.tight_layout()  # Keeps labels from overlapping
plt.show()