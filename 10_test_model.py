import torch
import numpy as np
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    average_precision_score, classification_report
)
from confidence_functions import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_DIR = '/gpfs/projects/b1042/AmaralLab/shin/DNN-Decision-Uncertainty-Quantification/models'
SIGMOID_PATH = f'{BASE_DIR}/sigmoid_resnet18_224x224_8ep.pt'
LABELS_PATH = f'{BASE_DIR}/test_labels_resnet18_224x224_8ep.pt'
NUM_CLASSES = 19
THRESHOLD = np.array([0.5] * NUM_CLASSES)  # change this for different decision thresholds of the sigmoid output


# Load saved probabilities and labels 
probs = torch.load(SIGMOID_PATH, map_location='cpu')['sigmoid'].numpy()      # (N, 19)
labels = torch.load(LABELS_PATH, map_location='cpu')['test_labels'].numpy()  # (N, 19)
 
print(f"Loaded {probs.shape[0]} test samples, {probs.shape[1]} classes")

THRESHOLD = find_best_threshold(probs, labels)
preds = (probs > THRESHOLD).astype(np.float32)


# Overall metrics
f1_macro = f1_score(labels, preds, average='macro', zero_division=0)
f1_micro = f1_score(labels, preds, average='micro', zero_division=0)
precision_macro = precision_score(labels, preds, average='macro', zero_division=0)
recall_macro = recall_score(labels, preds, average='macro', zero_division=0)
map_macro = average_precision_score(labels, probs, average='macro')
 
print("\n=== Overall Metrics ===")
print(f"Macro F1:        {f1_macro:.4f}")
print(f"Micro F1:        {f1_micro:.4f}")
print(f"Macro Precision: {precision_macro:.4f}")
print(f"Macro Recall:    {recall_macro:.4f}")
print(f"Macro mAP:       {map_macro:.4f}")



# Per-class metrics
f1_per_class = f1_score(labels, preds, average=None, zero_division=0)
precision_per_class = precision_score(labels, preds, average=None, zero_division=0)
recall_per_class = recall_score(labels, preds, average=None, zero_division=0)
 
print("\n=== Per-Class Metrics ===")
for c in range(NUM_CLASSES):
    support = int(labels[:, c].sum())
    print(f"Class {c:2d} | F1: {f1_per_class[c]:.3f} | "
          f"Precision: {precision_per_class[c]:.3f} | "
          f"Recall: {recall_per_class[c]:.3f} | Support: {support}")
 
print("\n=== Full classification_report ===")
print(classification_report(
    labels, preds,
    target_names=[f"class_{i}" for i in range(NUM_CLASSES)],
    zero_division=0
))
 
# Plot per-class F1
plt.figure(figsize=(10, 4))
plt.bar(range(NUM_CLASSES), f1_per_class, color='skyblue')
plt.xlabel('Class')
plt.ylabel('F1 Score')
plt.title('Per-Class F1 Score on Test Set')
plt.xticks(range(NUM_CLASSES))
plt.ylim(0, 1)
plt.savefig(f'{BASE_DIR}/per_class_f1.png')
print(f"\nSaved per-class F1 plot to {BASE_DIR}/per_class_f1.png")