import torch
import numpy as np
import confidence_functions as cf

BASE_DIR = '/gpfs/projects/b1042/AmaralLab/shin/DNN-Decision-Uncertainty-Quantification/models'
MODEL_TAG = 'resnet18_224x224_8ep'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("load stats and alpha...")
stats_maha = torch.load(f'{BASE_DIR}/stats_{MODEL_TAG}.pt', map_location='cpu', weights_only=False)['stats']
alpha = torch.load(f'{BASE_DIR}/alpha_adv_sig_{MODEL_TAG}.pt', map_location='cpu', weights_only=False)['alpha']


print("load model...")
model = torch.load(f'{BASE_DIR}/model_{MODEL_TAG}.pt', map_location='cpu', weights_only=False)['model']
model.to(device)
model.eval()

print("load images...")
combined_imgs = torch.load(f'{BASE_DIR}/combined_imgs_{MODEL_TAG}.pt', map_location='cpu', weights_only=False, mmap=True)['combined_imgs']
batch_size=500

print(f"\nTotal images: {combined_imgs[0].shape}, batch size: {batch_size}")
maha_combined_list = []
for i in range(0, combined_imgs.shape[0], batch_size):
    print(f"get maha conf for images in batch {i}")
    batch = combined_imgs[i:i + batch_size].to(device)
    maha_batch = cf.mahalanobis(batch, model, stats_maha, alpha, device=device)
    print(f"print conf of first image in batch {i}: conf = {maha_batch[0]}")
    maha_combined_list.append(maha_batch)
maha_combined_np = np.concatenate(maha_combined_list, axis=0)

print("Save maha conf...")
torch.save({'maha_combined_np': maha_combined_np}, f'{BASE_DIR}/maha_combined_np_alpha_adv_sig_{MODEL_TAG}.pt')
print("Saved!")