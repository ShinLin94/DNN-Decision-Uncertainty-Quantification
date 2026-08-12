"""
11_visualizing_confidence.py

Three-part analysis:

  PART A: Load saved test-set artifacts (prob, labels, threshold, stability) and
           compute per-sample F1 manually, then scatter it against (a) mean
           sigmoid prob among above-threshold classes, and (b) Shannon stability.

  PART B: Build a negative set for compute_alpha: 4000 CIFAR-10 + 4000 noise
           (=8000 OOD) concatenated with 8000 FGSM-perturbed in-distribution
           images (=16000 total negatives), and a positive set of 8000 random
           training images. Computes per-layer Mahalanobis scores (M_l) for
           both, fits alpha via compute_alpha, and saves it.

  PART C: Runs test set + 2000 CIFAR-10 (red) + 2000 noise (blue) through the
           model and produces three scatter plots: F1 (0 for CIFAR/noise) vs
           mean above-threshold sigmoid prob, vs Shannon stability (cats=1),
           and vs Mahalanobis confidence using the trained alpha.       
"""

import gc

import numpy as np
from pyparsing import line
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score as sk_f1_score
from scipy import stats
from scipy.stats import linregress
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import confidence_functions as cf

BASE_DIR = '/gpfs/projects/b1042/AmaralLab/shin/DNN-Decision-Uncertainty-Quantification/models'
CACHE_PATH = '/gpfs/projects/b1042/AmaralLab/shin/DNN-Decision-Uncertainty-Quantification/cached_data_224.pt'
MODEL_TAG = 'resnet18_224x224_8ep'
NUM_CLASSES = 19
CIFAR_ROOT = '/gpfs/projects/b1042/AmaralLab/shin/DNN-Decision-Uncertainty-Quantification/data'
WORKERS = 4  # for DataLoader num_workers

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
# DataLoader generator
g = torch.Generator()
g.manual_seed(SEED)

plt.rcParams.update({
    'font.size': 18,          # Global text size
    'axes.titlesize': 24,     # Plot title size
    'axes.labelsize': 18,     # X and Y label size
    'xtick.labelsize': 14,    # X-axis tick font size
    'ytick.labelsize': 14,    # Y-axis tick font size
    'legend.fontsize': 16     # Legend font size
})

# ---------------------------------------------------------------------------
# PART A: per-sample F1 vs saved sigmoid / stability
# ---------------------------------------------------------------------------

def compute_per_sample_f1(labels, preds, eps=1e-12):
    """
    Manual per-sample (a.k.a. 'samples'-averaged, but kept per-row) F1.
    sklearn's f1_score(..., average='samples') only returns one aggregate
    number across the dataset, not one value per sample -- this gives the
    per-row values that aggregate reduces over.
    Rows with no true positives AND no predicted positives get F1=0.0,
    matching sklearn's zero_division=0 convention (flip to 1.0 if you'd
    rather treat "correctly predicted nothing" as a perfect match).
    """
    labels = labels.astype(np.float32)
    preds = preds.astype(np.float32)

    tp = (preds * labels).sum(axis=1)
    fp = (preds * (1 - labels)).sum(axis=1)
    fn = ((1 - preds) * labels).sum(axis=1)

    denom = 2 * tp + fp + fn
    f1 = np.where(denom > 0, (2 * tp) / (denom + eps), 0.0)
    return f1


def scatter_plot(x, y, xlabel, ylabel, title, save_path, color='steelblue'):
    # get rrid of nan and inf in x and y
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    # print if there were any NaN or inf values removed
    n_removed = len(mask) - np.sum(mask)
    print(f"Removed {n_removed} samples with NaN or inf values from scatter plot.")

    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    line = slope * x + intercept
    print(f"Scatter plot: {title}")
    print(f"Slope: {slope:.2e}, R-value: {r_value:.2f}, R2: {r_value**2:.2f}, P-value: {p_value:.2e}")

    plt.figure(figsize=(6, 5))
    plt.scatter(x, y, s=8, alpha=0.4, color=color)
    plt.plot(x, line, color="red", linewidth=2, label="Line of Best Fit")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def scatter_plot_three_groups(x, y, n_test, n_adv, n_ood,
                               xlabel, ylabel, title, save_path):
    # get rrid of nan and inf in x and y
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    # print if there were any NaN or inf values removed
    n_removed = len(mask) - np.sum(mask)
    print(f"Removed {n_removed} samples with NaN or inf values from scatter plot.")

    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    line = slope * x + intercept
    print(f"Scatter plot: {title}")
    print(f"Slope: {slope:.4f}, R-value: {r_value:.4f}, R2: {r_value**2:.2f}, P-value: {p_value:.4f}")

    plt.figure(figsize=(6, 5))
    x_test, y_test = x[:n_test], y[:n_test]
    x_adv, y_adv = x[n_test:n_test + n_adv], y[n_test:n_test + n_adv]
    x_ood, y_ood = x[n_test + n_adv:], y[n_test + n_adv:]

    # plt.scatter(x_ood, y_ood, s=10, alpha=0.6, color='green', label='OOD (out-of-distribution)')
    plt.hexbin(x_ood, y_ood, bins=10, cmap='Blues', label='OOD (out-of-distribution)')
    plt.hexbin(x_adv, y_adv, bins=10, cmap='Blues', label='Adversarial')
    plt.hexbin(x_test, y_test, bins=10, cmap='Blues', label='Test (in-distribution)')

    plt.plot(x, line, color="red", linewidth=2, label="Line of Best Fit")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")

# make box plot for in distribution vs vs adversarial vs out of distribution for mean above threshold prob, stability, and mahalanobis
def box_plot_three_groups(x, n_test, n_adv, n_ood, xlabel, title, save_path):
    # get rrid of nan and inf in x
    mask = np.isfinite(x)
    x = x[mask]

    # print if there were any NaN or inf values removed
    n_removed = len(mask) - np.sum(mask)
    print(f"Removed {n_removed} samples with NaN or inf values from box plot.")

    plt.figure(figsize=(6, 5))
    x_test, x_adv, x_ood = x[:n_test], x[n_test:n_test + n_adv], x[n_test + n_adv:]

    plt.boxplot([x_test, x_adv, x_ood], tick_labels=['Test (in-distribution)', 'Adversarial', 'OOD (out-of-distribution)'])
    plt.xlabel(xlabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")

print("\n=== PART A: per-sample F1 vs saved sigmoid/stability ===")
print("Loading saved model, sigmoid probs, labels, stability, threshold, stats...")

model = torch.load(f'{BASE_DIR}/model_{MODEL_TAG}.pt', map_location='cpu', weights_only=False)['model']
model.to(device)
model.eval()

prob = torch.load(f'{BASE_DIR}/sigmoid_{MODEL_TAG}.pt', map_location='cpu', weights_only=False)['prob']
test_labels = torch.load(f'{BASE_DIR}/test_labels_{MODEL_TAG}.pt', map_location='cpu', weights_only=False)['test_labels']
stability = torch.load(f'{BASE_DIR}/stability_{MODEL_TAG}.pt', map_location='cpu', weights_only=False)['stability']
threshold = np.asarray(torch.load(f'{BASE_DIR}/threshold_{MODEL_TAG}.pt', map_location='cpu', weights_only=False)['threshold'])
stats = torch.load(f'{BASE_DIR}/stats_{MODEL_TAG}.pt', map_location='cpu', weights_only=False)['stats']

prob_np = prob.numpy() if torch.is_tensor(prob) else np.asarray(prob)
labels_np = test_labels.numpy() if torch.is_tensor(test_labels) else np.asarray(test_labels)
stability_np = stability.numpy() if torch.is_tensor(stability) else np.asarray(stability)

preds_np = (prob_np > threshold).astype(np.float32)

f1_per_sample = compute_per_sample_f1(labels_np, preds_np)

sanity = sk_f1_score(labels_np, preds_np, average='samples', zero_division=0)
print(f"Sanity check -- mean(manual per-sample F1) = {f1_per_sample.mean():.4f}, "
      f"sklearn average='samples' = {sanity:.4f} (should match closely)")

# mean sigmoid prob among classes ABOVE threshold, per sample
above_thresh_mask = prob_np > threshold
mean_prob_above = np.full(len(prob_np), np.nan)
for i in range(len(prob_np)):
    vals = prob_np[i][above_thresh_mask[i]]
    if len(vals) > 0:
        mean_prob_above[i] = vals.mean()
# samples where nothing exceeds threshold stay NaN; matplotlib skips NaNs in scatter

scatter_plot(
    f1_per_sample, mean_prob_above,
    xlabel="Per-sample F1", ylabel="Mean Sigmoid Prob Above Threshold",
    title="F1 vs Sigmoid Prob per Sample",
    save_path=f'{BASE_DIR}/f1_vs_mean_prob.png'
)

scatter_plot(
    f1_per_sample, stability_np,
    xlabel="Per-sample F1", ylabel="Mean Shannon Entropy Across",
    title="F1 vs Entropy per Sample",
    save_path=f'{BASE_DIR}/f1_vs_shannon.png'
)

del prob, test_labels, stability, preds_np, above_thresh_mask, mean_prob_above
gc.collect()  # free up memory before next part

# ---------------------------------------------------------------------------
# PART B: build OOD + adversarial negatives, train alpha
# ---------------------------------------------------------------------------

print("\n=== PART B: building negative/positive sets and training alpha ===")
print("Loading cached image data and train/test indices...")

cache = torch.load(CACHE_PATH, map_location='cpu', weights_only=False)
all_imgs, all_labels = cache['images'], cache['labels']

idx = torch.load(f'{BASE_DIR}/train_test_indices.pt', map_location='cpu', weights_only=False)
train_indices, test_indices = idx['train_indices'], idx['test_indices']

train_imgs_all = all_imgs[train_indices]
train_labels_all = all_labels[train_indices]
test_imgs_all = all_imgs[test_indices]
test_labels_all = all_labels[test_indices]
del all_imgs, all_labels, idx, cache
gc.collect()

print("Computing normalization stats from training set (chunked)...")
NORM_MEAN, NORM_STD = cf.compute_dataset_stats(train_imgs_all)
print(f"mean={NORM_MEAN}, std={NORM_STD}")

# Monkey-patch confidence_functions' module-level constants so
# get_cifar10_ood_loader / get_noise_ood_loader normalize identically to
# your real training data, regardless of whatever placeholder is currently
# hardcoded at the top of that file.
# NORM_MEAN = (0.077, 0.046, 0.075)
# NORM_STD = (0.135, 0.097, 0.169)
# IMG_SIZE = 224
cf.NORM_MEAN = NORM_MEAN
cf.NORM_STD = NORM_STD
cf.IMG_SIZE = 224

def normalize_uint8_batch(imgs_uint8, mean=NORM_MEAN, std=NORM_STD):
    """imgs_uint8: (N, 3, H, W) uint8 tensor -> normalized float tensor."""
    mean_t = torch.tensor(mean).view(1, -1, 1, 1)
    std_t = torch.tensor(std).view(1, -1, 1, 1)
    imgs = imgs_uint8.float() / 255.0
    return (imgs - mean_t) / std_t


print("Sampling 8000 positive (in-distribution training) examples...")
pos_sample_idx = torch.randperm(len(train_imgs_all), generator=g)[:8000]
pos_imgs = normalize_uint8_batch(train_imgs_all[pos_sample_idx])

print("Loading 4000 CIFAR-10 OOD images...")
# NOTE: CIFAR-10 download requires internet access. If this runs on a Quest
# compute node without outbound access, download once on the login node with
# download=True, then point CIFAR_ROOT here at that cached location and this
# will use the local copy without needing network access again.
cifar_loader = cf.get_cifar10_ood_loader(root=CIFAR_ROOT, batch_size=500, download=True, num_workers=WORKERS)
cifar_imgs_list, n_collected = [], 0
for imgs, _ in cifar_loader:
    cifar_imgs_list.append(imgs)
    n_collected += imgs.shape[0]
    if n_collected >= 2000:
        break
cifar_imgs = torch.cat(cifar_imgs_list, dim=0)[:2000]

print("Generating 4000 noise OOD images...")
noise_loader = cf.get_noise_ood_loader(n_samples=2000, batch_size=500, img_size=224, mode="gaussian", num_workers=WORKERS)
noise_imgs = torch.cat([imgs for imgs, _ in noise_loader], dim=0)[:2000]

ood_imgs = torch.cat([cifar_imgs, noise_imgs], dim=0)  # 4000
print(f"OOD negatives: {ood_imgs.shape[0]}")
del noise_imgs, cifar_imgs, cifar_imgs_list
gc.collect()

print("Generating 8000 FGSM adversarial examples from training data...")
adv_source_idx = torch.randperm(len(train_imgs_all), generator=g)[:4000]
adv_source_imgs = normalize_uint8_batch(train_imgs_all[adv_source_idx])
adv_source_labels = train_labels_all[adv_source_idx]

adv_loader = DataLoader(TensorDataset(adv_source_imgs, adv_source_labels),
                         batch_size=500, shuffle=False, num_workers=WORKERS)
adv_imgs_list = []
for imgs, labels in adv_loader:
    adv_batch = cf.fgsm_attack(model, imgs, labels, epsilon=0.02, device=device)
    adv_imgs_list.append(adv_batch)
adv_imgs = torch.cat(adv_imgs_list, dim=0)[:4000]
print(f"Adversarial negatives: {adv_imgs.shape[0]}")

neg_imgs = torch.cat([ood_imgs, adv_imgs], dim=0)  # 8000
print(f"Total negatives: {neg_imgs.shape[0]} (4000 OOD + 4000 adversarial)")
print(f"Total positives: {pos_imgs.shape[0]}")
del train_imgs_all, train_labels_all, adv_source_imgs, adv_source_labels, adv_imgs, ood_imgs
gc.collect()  # free up memory before next part

print("Fitting alpha via compute_alpha...")
alpha = cf.compute_alpha(model, pos_imgs, neg_imgs, stats, device=device)
torch.save({'alpha': alpha}, f'{BASE_DIR}/alpha_{MODEL_TAG}.pt')
print(f"Saved trained alpha: {alpha}")
del pos_imgs, neg_imgs
gc.collect()

# print("\n=== PART B.2: loading trained alpha and stats for Mahalanobis confidence ===")
# alpha = torch.load(f'{BASE_DIR}/alpha_{MODEL_TAG}.pt', map_location='cpu', weights_only=False)['alpha']

# ---------------------------------------------------------------------------
# PART C: combined test + CIFAR-10 (red) + noise (blue) scatter plots
# ---------------------------------------------------------------------------

print("\n=== PART C: combined test + CIFAR-10 + noise scatter plots ===")

print("Loading 2000 CIFAR-10 images for combined set...")
cifar_loader_2k = cf.get_cifar10_ood_loader(root=CIFAR_ROOT, batch_size=500, download=True, num_workers=WORKERS)
cifar_2k_list, n_collected = [], 0
for imgs, _ in cifar_loader_2k:
    cifar_2k_list.append(imgs)
    n_collected += imgs.shape[0]
    if n_collected >= 2000:
        break
cifar_2k = torch.cat(cifar_2k_list, dim=0)[:2000]
n_cifar = cifar_2k.shape[0]

print("Generating 2000 noise images for combined set...")
noise_loader_2k = cf.get_noise_ood_loader(n_samples=2000, batch_size=500, img_size=224, mode="gaussian", num_workers=WORKERS)
noise_2k = torch.cat([imgs for imgs, _ in noise_loader_2k], dim=0)[:2000]
n_noise = noise_2k.shape[0]

print("Generating 2000 FGSM adversarial examples for combined set...")
adv_source_idx_2k = torch.randperm(len(test_imgs_all), generator=g)[:2000]
adv_source_imgs_2k = normalize_uint8_batch(test_imgs_all[adv_source_idx_2k])
adv_source_labels_2k = test_labels_all[adv_source_idx_2k]
n_adv = adv_source_imgs_2k.shape[0]

adv_loader = DataLoader(TensorDataset(adv_source_imgs_2k, adv_source_labels_2k),
                         batch_size=500, shuffle=False, num_workers=WORKERS)
adv_imgs_list = []
for imgs, labels in adv_loader:
    adv_batch = cf.fgsm_attack(model, imgs, labels, epsilon=0.02, device=device)
    adv_imgs_list.append(adv_batch)
adv_imgs_2k = torch.cat(adv_imgs_list, dim=0)[:8000]

test_imgs_norm = normalize_uint8_batch(test_imgs_all)
n_test = test_imgs_norm.shape[0]

del test_imgs_all, adv_source_imgs_2k, adv_imgs_list, cifar_2k_list
gc.collect()

combined_imgs = torch.cat([test_imgs_norm, adv_imgs_2k, cifar_2k, noise_2k], dim=0)
print(f"Combined set: {n_test} test + {n_adv} adversarial + {n_cifar} CIFAR-10 + {n_noise} noise = {combined_imgs.shape[0]}")
del adv_imgs_2k, cifar_2k, noise_2k
gc.collect()  # free up memory before next part
# save combined images
torch.save({'combined_imgs': combined_imgs}, f'{BASE_DIR}/combined_imgs_{MODEL_TAG}.pt')
# combined_imgs = torch.load(f'{BASE_DIR}/combined_imgs_{MODEL_TAG}.pt', map_location='cpu', weights_only=False)['combined_imgs']

print("Running combined set through model for sigmoid probabilities...")
batch_size = 1000
combined_probs_list = []
with torch.no_grad():
    model.eval()
    for i in range(0, combined_imgs.shape[0], batch_size):
        batch = combined_imgs[i:i + batch_size].to(device)
        logits = model(batch)
        combined_probs_list.append(torch.sigmoid(logits).cpu())
combined_probs = torch.cat(combined_probs_list, dim=0)
combined_probs_np = combined_probs.numpy()


print("Computing per-sample F1 on combined set...")
# pred_test_adv = (combined_probs_np[:n_test+n_adv] > threshold).astype(np.float32)
# label_test_adv = np.concatenate([test_labels_all.numpy(), adv_source_labels_2k.numpy()])
# f1_combined = np.concatenate([compute_per_sample_f1(label_test_adv, pred_test_adv), np.zeros(n_cifar), np.zeros(n_noise)])
# del pred_test_adv, label_test_adv, combined_probs_list, adv_source_labels_2k, test_labels_all

predictions = (combined_probs_np > threshold).astype(np.float32)
labels_ood = np.zeros((n_cifar + n_noise, NUM_CLASSES), dtype=np.float32)
labels_ood[:, -1] = 1.0
labels = np.concatenate([test_labels_all.numpy(), adv_source_labels_2k.numpy(), labels_ood])
f1_combined = compute_per_sample_f1(labels, predictions)
del predictions, labels_ood, labels, combined_probs_list, adv_source_labels_2k, test_labels_all
torch.save({'f1_combined': f1_combined}, f'{BASE_DIR}/f1_combined_{MODEL_TAG}.pt')
# f1_combined = torch.load(f'{BASE_DIR}/f1_combined_{MODEL_TAG}.pt', map_location='cpu', weights_only=False)['f1_combined']


# save combine images, combine probs, 

print("Get mean sigmoid prob among above-threshold classes, per sample...")
above_mask = combined_probs_np > threshold
mean_prob_combined = np.full(len(combined_probs_np), np.nan)
for i in range(len(combined_probs_np)):
    vals = combined_probs_np[i][above_mask[i]]
    if len(vals) > 0:
        mean_prob_combined[i] = vals.mean()
torch.save({'mean_prob_combined': mean_prob_combined}, f'{BASE_DIR}/mean_prob_combined_{MODEL_TAG}.pt')
# mean_prob_combined = torch.load(f'{BASE_DIR}/mean_prob_combined_{MODEL_TAG}.pt', map_location='cpu', weights_only=False)['mean_prob_combined']


print("Computing Shannon stability on combined set (cats=1)...")
# Per your instruction: feed the sigmoid probs directly into shannon_stability_multilabel
# with cats=1, rather than the usual accumulated-decision-count cat_all.
stability_combined = cf.shannon_stability_multilabel(combined_probs.to(device), 1)
stability_combined_np = (stability_combined.cpu().numpy() if torch.is_tensor(stability_combined)
                          else np.asarray(stability_combined))
torch.save({'stability_combined_np': stability_combined_np}, f'{BASE_DIR}/stability_combined_np_{MODEL_TAG}.pt')
# stability_combined_np = torch.load(f'{BASE_DIR}/stability_combined_np_{MODEL_TAG}.pt', map_location='cpu', weights_only=False)['stability_combined_np']

print("Computing Mahalanobis confidence (trained alpha) on combined set...")
# NOTE: cf.mahalanobis() internally calls get_features(model, X) -- this will
# only work if that resolves to the ResNet-compatible extractor (see
# prerequisite #2 in the module docstring).
maha_combined_list = []
for i in range(0, combined_imgs.shape[0], batch_size):
    batch = combined_imgs[i:i + batch_size].to(device)
    maha_batch = cf.mahalanobis(batch, model, stats, alpha, device=device)
    maha_combined_list.append(maha_batch)
maha_combined_np = np.concatenate(maha_combined_list, axis=0)
torch.save({'maha_combined_np': maha_combined_np}, f'{BASE_DIR}/maha_combined_np_{MODEL_TAG}.pt')
# maha_combined_np = torch.load(f'{BASE_DIR}/maha_combined_np_{MODEL_TAG}.pt', map_location='cpu', weights_only=False)['maha_combined_np']

del combined_imgs, combined_probs, above_mask, maha_combined_list, stability_combined
gc.collect()

n_ood=n_cifar+n_noise
scatter_plot_three_groups(
    f1_combined, mean_prob_combined, n_test, n_adv, n_ood,
    xlabel="F1 per Sample", ylabel="Mean Sigmoid Prob Above Threshold",
    title="F1 vs Sigmoid Prob",
    save_path=f'{BASE_DIR}/f1_vs_mean_prob_combined.png'
)

scatter_plot_three_groups(
    f1_combined, stability_combined_np, n_test, n_adv, n_ood,
    xlabel="F1 per Sample", ylabel="Entropy from Sigmoid Prob",
    title="F1 vs Entropy",
    save_path=f'{BASE_DIR}/f1_vs_stability_combined.png'
)

scatter_plot_three_groups(
    f1_combined, maha_combined_np, n_test, n_adv, n_ood,
    xlabel="F1 per Sample", ylabel="Mahalanobis Confidence",
    title="F1 vs Mahalanobis Confidence",
    save_path=f'{BASE_DIR}/f1_vs_mahalanobis_combined.png'
)

box_plot_three_groups(
    mean_prob_combined, n_test, n_adv, n_ood,
    xlabel="Sigmoid Probabilities",
    title="Mean Sigmoid Probability Above Threshold",
    save_path=f'{BASE_DIR}/box_mean_prob_combined.png'
)

box_plot_three_groups(
    stability_combined_np, n_test, n_adv, n_ood,
    xlabel="Shannon Entropy",
    title="Entropy using Sigmoid Prob",
    save_path=f'{BASE_DIR}/box_stability_combined.png'
)

box_plot_three_groups(
    maha_combined_np, n_test, n_adv, n_ood,
    xlabel="Confidence Score",
    title="Mahalanobis Confidence",
    save_path=f'{BASE_DIR}/box_mahalanobis_combined.png'
)


import pandas as pd
import seaborn as sns

# Helper function to remove NaN/Inf and slice groups
def clean_and_split(x, n_test, n_adv, n_ood):
    mask = np.isfinite(x)
    x = x[mask]
    return x[:n_test], x[n_test:n_test + n_adv], x[n_test + n_adv:]

group_labels = (['Test (in-distribution)'] * n_test + 
                ['Adversarial'] * n_adv + 
                ['OOD (out-of-distribution)'] * n_ood)

p_test, p_adv, p_ood = clean_and_split(mean_prob_combined, n_test, n_adv, n_ood)
s_test, s_adv, s_ood = clean_and_split(stability_combined_np, n_test, n_adv, n_ood)
m_test, m_adv, m_ood = clean_and_split(maha_combined_np, n_test, n_adv, n_ood)

df = pd.DataFrame({
    'Group': group_labels,
    'Mean Probability': np.concatenate([p_test, p_adv, p_ood]),
    'Shannon Stability': np.concatenate([s_test, s_adv, s_ood]),
    'Mahalanobis Confidence': np.concatenate([m_test, m_adv, m_ood])
})

# Reshape into Tidy Format
df_melted = pd.melt(df, id_vars=['Group'], var_name='Metric', value_name='Value')

palette = {'Test (in-distribution)': '#2b5c8f', 'Adversarial': '#d95f02', 'OOD (out-of-distribution)': '#7570b3'}

g = sns.catplot(
    data=df_melted,
    x='Methods',
    y='Confidence',
    col='Metric',
    hue='Data Group',
    kind='box',
    palette=palette,
    sharey=False,  # Allows independent Y-scales per metric
    height=4,
    aspect=1.0,
    dodge=False
)

g.set_xticklabels(rotation=15)
g.set_titles(col_template="{col_name}")
plt.tight_layout()
plt.savefig(f'{BASE_DIR}/combined_box_plots_subplots.png', dpi=150)



# everything on a single axis for easier comparison of distributions across metrics
plt.figure(figsize=(10, 6))
sns.boxplot(
    data=df_melted,
    x='Metric',
    y='Value',
    hue='Group',
    palette=['#2b5c8f', '#d95f02', '#7570b3']
)

plt.title('Distribution Comparison across Metrics', fontsize=24)
plt.xlabel('Metric', fontsize=18)
plt.ylabel('Value', fontsize=18)
plt.legend(title='Dataset Group', frameon=True, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(f'{BASE_DIR}/combined_box_plots_single_axis.png', dpi=150)

print("\nAll done. Plots saved to:", BASE_DIR)