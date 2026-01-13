import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.data import Dataset


import os
import numpy as np
import nibabel as nib
import random
import glob

import random
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF

class MultiModalMRISSLTransform:
    """
    Optimized for MPFlow (DPS-style Flow Matching):
    - Conservative shared geometry to maintain spatial equivariance and latent precision.
    - Aggressive, independent intensity jitter to force contrast-invariant structural learning.
    """
    def __init__(self,
                 # Tighter geometric ranges to avoid interpolation-induced blurring
                 max_rotation=3.0,        # Reduced from 10 to preserve grid alignment
                 max_translate=0.02,      # Reduced from 0.1 to keep patches centered
                 scale_range=(0.98, 1.02), # Tighter scale for sub-pixel precision
                 
                 blur_prob=0.2,           # Lower probability
                 blur_sigma_range=(0.1, 0.5), # Light blur only to keep edges sharp
                 
                 dsus_prob=0.2,           # Lower probability for SR tasks
                 ds_factor_range=(1, 2),  # Minimal downsampling
                 
                 gamma_prob=0.8,          # High probability for contrast invariance
                 gamma_range=(0.5, 1.5),  # Wider range for style-agnosticism
                 
                 noise_prob=0.5,
                 noise_std=0.01,          # Moderate noise for manifold smoothing
                 
                 sharp_prob=0.2,
                 sharp_factor=1.2):
        
        self.max_rotation = max_rotation
        self.max_translate = max_translate
        self.scale_range = scale_range
        self.blur_prob = blur_prob
        self.blur_sigma_range = blur_sigma_range
        self.dsus_prob = dsus_prob
        self.ds_factor_range = ds_factor_range
        self.gamma_prob = gamma_prob
        self.gamma_range = gamma_range
        self.noise_prob = noise_prob
        self.noise_std = noise_std
        self.sharp_prob = sharp_prob
        self.sharp_factor = sharp_factor

    def _shared_geom_and_res(self, img1, img2):
        # Shared geometry is CRITICAL for DenseInfoNCE spatial correspondence
        _, H, W = img1.shape

        # --- shared affine ---
        angle = random.uniform(-self.max_rotation, self.max_rotation)
        tx = random.uniform(-self.max_translate * W, self.max_translate * W)
        ty = random.uniform(-self.max_translate * H, self.max_translate * H)
        scale = random.uniform(*self.scale_range)

        def apply_affine(img):
            return TF.affine(
                img,
                angle=angle,
                translate=[tx, ty],
                scale=scale,
                shear=[0.0, 0.0],
                interpolation=T.InterpolationMode.BILINEAR # Best trade-off for small angles
            )

        img1 = apply_affine(img1)
        img2 = apply_affine(img2)

        # --- shared downsample / upsample ---
        if random.random() < self.dsus_prob:
            factor = random.randint(self.ds_factor_range[0], self.ds_factor_range[1])
            new_size = (H // factor, W // factor)
            img1 = T.Resize(new_size, antialias=True)(img1)
            img1 = T.Resize((H, W), antialias=True)(img1)
            img2 = T.Resize(new_size, antialias=True)(img2)
            img2 = T.Resize((H, W), antialias=True)(img2)

        # --- shared blur ---
        if random.random() < self.blur_prob:
            sigma = random.uniform(*self.blur_sigma_range)
            blur = T.GaussianBlur(kernel_size=5, sigma=sigma)
            img1 = blur(img1)
            img2 = blur(img2)

        return img1, img2

    def _intensity_jitter(self, img):
        # INDEPENDENT intensity transforms force the encoders to ignore "shortcuts"
        
        # 1. Aggressive Gamma (Contrast Invariance)
        if random.random() < self.gamma_prob:
            gamma = random.uniform(*self.gamma_range)
            img = torch.clamp(img, 1e-6, 1.0) ** gamma

        # 2. Random Sharpness
        if random.random() < self.sharp_prob:
            # Note: T.RandomAdjustSharpness is p-driven; we apply it directly here
            img = TF.adjust_sharpness(img, self.sharp_factor)

        # 3. Additive Noise (Robustness to measurement noise)
        if random.random() < self.noise_prob:
            img = img + torch.randn_like(img) * self.noise_std

        return torch.clamp(img, 0.0, 1.0)

    def __call__(self, img_t1, img_t2):
        # Step 1: Shared geometry ensures anatomical landmarks are in the same relative pixel locations
        img_t1, img_t2 = self._shared_geom_and_res(img_t1, img_t2)

        # Step 2: Independent intensity ensures encoders learn "Structure" rather than "Pixel Values"
        img_t1 = self._intensity_jitter(img_t1)
        img_t2 = self._intensity_jitter(img_t2)

        return img_t1, img_t2

class MultiModalMRISSLTransform_OLD:
    """
    Option A:
    - Shared geometry / resolution / blur for T1 & T2
    - Modality-specific intensity jitter (gamma, noise, sharpness)
    Assumes inputs are tensors of shape (C, H, W) in [0, 1].
    """
    def __init__(self,
                 max_rotation=10,
                 max_translate=0.10,     # as fraction of H,W
                 scale_range=(0.95, 1.05),
                 blur_prob=0.5,
                 blur_sigma_range=(0.1, 1.0),
                 dsus_prob=0.5,
                 ds_factor_range=(2, 4),
                 gamma_prob=0.5,
                 gamma_range=(0.7, 1.3),
                 noise_prob=0.5,
                 noise_std=0.05,
                 sharp_prob=0.25,
                 sharp_factor=1.5):
        self.max_rotation = max_rotation
        self.max_translate = max_translate
        self.scale_range = scale_range
        self.blur_prob = blur_prob
        self.blur_sigma_range = blur_sigma_range
        self.dsus_prob = dsus_prob
        self.ds_factor_range = ds_factor_range
        self.gamma_prob = gamma_prob
        self.gamma_range = gamma_range
        self.noise_prob = noise_prob
        self.noise_std = noise_std
        self.sharp_prob = sharp_prob
        self.sharp_factor = sharp_factor

    def _shared_geom_and_res(self, img1, img2):
        # img1, img2: (C, H, W)
        _, H, W = img1.shape

        # --- shared affine (rotation + translation + scale) ---
        angle = random.uniform(-self.max_rotation, self.max_rotation)

        max_tx = self.max_translate * W
        max_ty = self.max_translate * H
        tx = random.uniform(-max_tx, max_tx)
        ty = random.uniform(-max_ty, max_ty)

        scale = random.uniform(*self.scale_range)
        shear = [0.0, 0.0]

        def apply_affine(img):
            return TF.affine(
                img,
                angle=angle,
                translate=[tx, ty],
                scale=scale,
                shear=shear,
                interpolation=T.InterpolationMode.BILINEAR
            )

        img1 = apply_affine(img1)
        img2 = apply_affine(img2)

        # --- shared downsample / upsample ---
        if random.random() < self.dsus_prob:
            factor = random.randint(self.ds_factor_range[0], self.ds_factor_range[1])
            new_H, new_W = H // factor, W // factor
            ds = T.Resize((new_H, new_W), antialias=True)
            us = T.Resize((H, W), antialias=True)
            img1 = us(ds(img1))
            img2 = us(ds(img2))

        # --- shared blur ---
        if random.random() < self.blur_prob:
            sigma = random.uniform(*self.blur_sigma_range)
            blur = T.GaussianBlur(kernel_size=5, sigma=sigma)
            img1 = blur(img1)
            img2 = blur(img2)

        return img1, img2

    def _intensity_jitter(self, img):
        # optional gamma
        if random.random() < self.gamma_prob:
            gamma = random.uniform(*self.gamma_range)
            # clamp to avoid 0 ** gamma issues
            img = torch.clamp(img, 1e-6, 1.0) ** gamma

        # optional sharpness (treated as intensity-like op here)
        if random.random() < self.sharp_prob:
            sharp = T.RandomAdjustSharpness(self.sharp_factor, p=1.0)
            img = sharp(img)

        # optional noise
        if random.random() < self.noise_prob:
            img = img + torch.randn_like(img) * self.noise_std

        img = torch.clamp(img, 0.0, 1.0)
        return img

    def __call__(self, img_t1, img_t2):
        # 1) shared geometry / res / blur
        img_t1, img_t2 = self._shared_geom_and_res(img_t1, img_t2)

        # 2) modality-specific intensity jitter
        img_t1 = self._intensity_jitter(img_t1)
        img_t2 = self._intensity_jitter(img_t2)

        return img_t1, img_t2


class IQTDataset(Dataset):
    def __init__(self, files_t1, configs, slice_idx=(100, 150, 2), return_id=False, transform=None, patch=False, train=True): #100 150 5
        super().__init__()
        
        self.files = files_t1
        self.slice_idx = slice_idx
        self.return_id = return_id
        self.configs = configs
        self.train = train
        self.patch = patch
        self.transform = MultiModalMRISSLTransform() if transform is not None else None
        if (self.patch is True) and (transform is not None):
            self.transform = MultiModalMRISSLTransform(
                 max_rotation=3.0,        # Reduced from 10 to preserve grid alignment
                 max_translate=0.02,      # Reduced from 0.1 to keep patches centered
                 scale_range=(0.98, 1.02), # Tighter scale for sub-pixel precision
                 blur_prob=0.2,           # Lower probability
                 blur_sigma_range=(0.1, 0.5), # Light blur only to keep edges sharp
                 dsus_prob=0.2,           # Lower probability for SR tasks
                 ds_factor_range=(1, 2),  # Minimal downsampling
                 gamma_prob=0.8,          # High probability for contrast invariance
                 gamma_range=(0.5, 1.5),  # Wider range for style-agnosticism
                 noise_prob=0.5,
                 noise_std=0.005,          # Moderate noise for manifold smoothing
                 sharp_prob=0.2,
                 sharp_factor=1.2)
                             #max_rotation=5,
                             #max_translate=0.01,     # as fraction of H,W
                             #scale_range=(0.9, 1.1),
                             #blur_prob=0.5,
                             #blur_sigma_range=(0.1, 1.0),
                             #dsus_prob=0.5,
                             #ds_factor_range=(1,2),
                             #gamma_prob=0.5,
                             #gamma_range=(0.8, 1.2),
                             #noise_prob=0.5,
                             #noise_std=0.005,
                             #sharp_prob=0.25,
                             #sharp_factor=1.1)
        self.lst = []
        for file in self.files:
            img_t1 = nib.load(file).get_fdata()
            file_t2 = file.replace('T1w_acpc', 'T2w_acpc')
            img_t2 = nib.load(file_t2).get_fdata()
        
            if self.return_id:
                file_id = file.split('/')[-3]
                file_id = int(file_id)
                
                for i in range(self.slice_idx[0], self.slice_idx[1], self.slice_idx[2]):
                    self.lst.append([img_t1[:,:,i], img_t2[:,:,i], file_id, i])
            else:
                for i in range(self.slice_idx[0], self.slice_idx[1], self.slice_idx[2]):
                    self.lst.append([img_t1[:,:,i], img_t2[:,:,i]])
                
    def cube(self,data):

        hyp_norm = data

        if len(hyp_norm.shape)>3:
            hyp_norm = hyp_norm[:,:, 2:258, 27:283]
        else:
            hyp_norm = hyp_norm[2:258, 27:283]

        return hyp_norm

    def __len__(self):
        return len(self.lst)

    def normalize(self, arr):
        if self.configs['norm'] == 'minmax':
            arr = arr/4096.0
        elif self.configs['norm'] == 'zscore':
            arr = (arr - self.configs['Data']['mean_hr'])/self.configs['Data']['std_hr']
        else: 
            # arr = arr/4096.0
            arr = 2*arr
        return arr
    
    # def transform(self, img):
    #     transform = transforms.Compose([
    #         transforms.RandomRotation(degrees=(0, 15)),  
    #         transforms.RandomAffine(degrees=0, translate=(0.0, 0.1), scale=(0.9, 1.1)),
    #         transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 1.0)),
    #         transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.25),
    #     ])
    #     random_ds_factor = np.random.randint(2, 5)
    #     random_ds = transforms.Resize((img.shape[0]//random_ds_factor, img.shape[1]//random_ds_factor))
    #     random_us = transforms.Resize((img.shape[0], img.shape[1]))
    #     random_gamma = lambda x: x ** np.random.uniform(0.7, 1.3)
    #     img = transform(img)
    #     if random.random() < 0.5:
    #         img = random_ds(img)
    #         img = random_us(img)
    #     if random.random() < 0.5:
    #         img = random_gamma(img)
    #     if random.random() < 0.5:
    #         noise = torch.randn(img.size()) * 0.05
    #         img = img + noise
    #         img = torch.clamp(img, 0.0, 1.0)
    #     return img

    def __getitem__(self, idx):
        if self.return_id:
            img_t1, img_t2, file_id, slice_idx = self.lst[idx]
        else:
            img_t1, img_t2 = self.lst[idx]
            file_id = None
            slice_idx = None
        self.dict = {}
        self.dict['slice_idx'] = slice_idx
        self.dict['file_id'] = file_id
        
        if img_t1.shape != (256, 256):
            img_t1 = self.cube(img_t1)
            img_t2 = self.cube(img_t2)

        if self.train:
            img_t1 = img_t1 / 4096.0
            img_t2 = img_t2 / 4096.0

            #Convert to tensor
            img_t1 = torch.tensor(img_t1)
            img_t2 = torch.tensor(img_t2)
            if self.patch:
                background_mask = (img_t2 == 0.0)
                background_mask = background_mask.unsqueeze(0)
            #Transform
            if (self.transform is not None) and (random.random() > 0.2):
                img_t1, img_t2 = self.transform(img_t1.unsqueeze(0), img_t2.unsqueeze(0)) #Shape: (1, H, W)

            img_t1 = self.normalize(img_t1)
            img_t2 = self.normalize(img_t2)

            #Double  type
            img_t1 = img_t1.type(torch.float32)
            img_t2 = img_t2.type(torch.float32)
            if img_t1.shape[0] != 1:
                img_t1 = img_t1.unsqueeze(0)
                img_t2 = img_t2.unsqueeze(0)
            
            #assert img_t1.shape == (1, 256, 256), f"Shape is {img_t1.shape}"
            #assert img_t2.shape == (1, 256, 256), f"Shape is {img_t2.shape}"
        else:
            img_t1 = 2* (img_t1 / 4096.0)
            img_t2 = 2* (img_t2 / 4096.0)
           
            img_t1 = torch.tensor(img_t1).unsqueeze(0).type(torch.float32)
            img_t2 = torch.tensor(img_t2).unsqueeze(0).type(torch.float32)
            if self.patch:
                background_mask = (img_t2 == 0.0)
                #background_mask = background_mask.unsqueeze(0)

        # select random indicies between 0 and 256 patch size 64
        if self.patch:
            ps = 32
            while True:
                rand_ind = np.random.randint(0, 256-ps, size=2)
                img_t2_clean = img_t2.clone()
                img_t2_clean[background_mask > 0] = 0.0
                patch_t2 = img_t2_clean[0, rand_ind[0]:rand_ind[0]+ps, rand_ind[1]:rand_ind[1]+ps]
                if torch.count_nonzero(patch_t2)/(ps*ps) > 0.5:
                    break
            img_t1 = img_t1[:, rand_ind[0]:rand_ind[0]+ps, rand_ind[1]:rand_ind[1]+ps]
            img_t2 = img_t2[:, rand_ind[0]:rand_ind[0]+ps, rand_ind[1]:rand_ind[1]+ps]
        
        
        if self.return_id:
            return img_t1, img_t2, self.dict
        self.dict = {}
        return img_t1, img_t2, self.dict
    
class SimCLRModel(nn.Module):
    """
    SimCLR encoder + projection head (for InfoNCE) + lightweight decoder head (for reconstruction).
    - Use z for contrastive.
    - Decode from h (pre-projection) to avoid forcing the projection head to carry pixel-level detail.
    """
    def __init__(self, projection_dim=128, return_z=True, recon_ps=32, decoder=True):
        super().__init__()

        self.decoder = decoder
        base_model = models.resnet18(weights=None)
        base_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_ftrs = base_model.fc.in_features
        base_model.fc = nn.Identity()
        self.encoder = base_model

        self.projection_head = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Linear(512, projection_dim),
        )
        # Minimal decoder: map global feature h -> patch reconstruction (ps x ps).
        # This assumes you train on patches (e.g., 32x32), which matches your DPS-SSL usage.
        if self.decoder:
            self.recon_head = nn.Sequential(
                nn.Linear(num_ftrs, 512),
                nn.ReLU(),
                nn.Linear(512, recon_ps * recon_ps),
            )
        self.recon_ps = recon_ps
        self.return_z = return_z

    def forward(self, x):
        h = self.encoder(x)                    # (N, F)
        z = self.projection_head(h)            # (N, D)
        if self.decoder:
            x_hat = self.recon_head(h).view(-1, 1, self.recon_ps, self.recon_ps)  # (N,1,ps,ps)
        else:
            x_hat = None
        if self.return_z:
            return z, x_hat
        return h, x_hat
    
def info_nce_multimodal(z_list, temperature=0.1):
    """
    Multi-view / multi-modal InfoNCE (SimCLR-style) loss.

    Args:
        z_list: list of tensors, each of shape (N, D)
                e.g. [z_t1, z_t2] or [z_t1_v1, z_t1_v2, z_t2_v1, z_t2_v2]
        temperature: scalar float

    Returns:
        loss: scalar tensor
    """
    # Number of views (modalities * augmentations)
    n_views = len(z_list)
    assert n_views >= 2, "Need at least two views for InfoNCE."

    # Check batch sizes
    batch_size = z_list[0].shape[0]
    for z in z_list:
        assert z.shape[0] == batch_size, "All views must have same batch size."

    device = z_list[0].device

    # (n_views * N, D)
    features = torch.cat(z_list, dim=0)
    features = F.normalize(features, dim=1)

    # Labels: [0,1,...,N-1, 0,1,...,N-1, ...] length = n_views * N
    labels = torch.arange(batch_size, device=device)
    labels = labels.repeat(n_views)  # (n_views * N,)

    # Build mask of positives (same underlying index) vs negatives
    # label_matrix[i, j] = 1 if i and j are positives
    label_matrix = (labels.unsqueeze(0) == labels.unsqueeze(1))  # (B', B')
    # B' = n_views * N
    B = label_matrix.shape[0]

    # Similarity matrix
    similarity_matrix = torch.matmul(features, features.T)  # (B, B)

    # Remove self-comparisons on the diagonal
    self_mask = torch.eye(B, dtype=torch.bool, device=device)
    label_matrix = label_matrix[~self_mask].view(B, -1)          # (B, B-1)
    similarity_matrix = similarity_matrix[~self_mask].view(B, -1)

    # Positives: same label (other views of same sample)
    positives = similarity_matrix[label_matrix].view(B, -1)      # (B, n_pos)
    # Negatives: different labels (all other samples)
    negatives = similarity_matrix[~label_matrix].view(B, -1)     # (B, n_neg)

    # Concatenate positives and negatives into logits
    # For each anchor row:
    #   logits[i] = [sim(i, pos_1), ..., sim(i, neg_1), ...]
    logits = torch.cat([positives, negatives], dim=1)            # (B, n_pos + n_neg)

    # Targets: index 0 is always the "correct" positive (we just group positives first)
    targets = torch.zeros(B, dtype=torch.long, device=device)

    # Temperature scaling
    logits = logits / temperature

    loss = F.cross_entropy(logits, targets)
    return loss

class DenseSimCLRModel(nn.Module):
    def __init__(self, projection_dim=128, recon_ps=32, decoder=True):
        super().__init__()
        
        # 1. Backbone: Extract features before the GAP layer
        resnet = models.resnet18(weights=None)
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Everything up to the final conv layer (layer4)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2]) 
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        num_ftrs = 512 # For ResNet18 layer4

        # 2. GLOBAL Head (Standard SimCLR)
        self.global_projector = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Linear(512, projection_dim)
        )

        # 3. DENSE Head (DenseCL Novelty)
        # We use 1x1 convolutions to project the spatial grid (B, 512, H/32, W/32)
        self.dense_projector = nn.Sequential(
            nn.Conv2d(num_ftrs, 512, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(512, projection_dim, kernel_size=1)
        )

        # 4. Reconstruction Head (h -> pixels)
        self.decoder = decoder
        if self.decoder:
            self.recon_head = nn.Sequential(
                nn.Linear(num_ftrs, 512),
                nn.ReLU(),
                nn.Linear(512, recon_ps * recon_ps),
            )
        self.recon_ps = recon_ps

    def forward(self, x):
        # Extract feature map: (B, 512, S, S) where S is spatial size (e.g. 8)
        feat_map = self.backbone(x)
        
        # Global Branch
        h_global = self.avgpool(feat_map).view(feat_map.size(0), -1)
        z_global = self.global_projector(h_global)
        
        # Dense Branch: (B, D, S, S)
        z_dense = self.dense_projector(feat_map)
        
        # Reconstruction Branch
        x_hat = self.recon_head(h_global).view(-1, 1, self.recon_ps, self.recon_ps) if self.decoder else None
            
        return z_global, z_dense, x_hat
    
def global_info_nce_loss(z_i, z_j, temperature=0.1):
    """
    Standard InfoNCE for global vectors. 
    z_i: T1 features, z_j: T2 features
    """
    z_i = F.normalize(z_i, dim=1)
    z_j = F.normalize(z_j, dim=1)
    
    # Cosine similarity between all T1s and all T2s in the batch
    logits = torch.matmul(z_i, z_j.T) / temperature
    labels = torch.arange(z_i.size(0)).to(z_i.device)
    
    return F.cross_entropy(logits, labels)

def dense_info_nce_loss(z_dense_i, z_dense_j, temperature=0.1):
    """
    Spatial InfoNCE aligning the grids of T1 and T2.
    """
    B, D, S, S = z_dense_i.shape
    # Flatten spatial grid into 'local samples'
    queries = F.normalize(z_dense_i.permute(0, 2, 3, 1).reshape(-1, D), dim=1)
    keys = F.normalize(z_dense_j.permute(0, 2, 3, 1).reshape(-1, D), dim=1)
    
    logits = torch.matmul(queries, keys.T) / temperature
    labels = torch.arange(logits.size(0)).to(z_dense_i.device)
    
    return F.cross_entropy(logits, labels)

class MultiModalSSL(torch.nn.Module):
    def __init__(self, feats=128, return_z=True, decoder=True, dense=False):
        super().__init__()

        if dense:
            self.enc_t1 = DenseSimCLRModel(projection_dim=feats, recon_ps=32, decoder=decoder)
            self.enc_t2 = DenseSimCLRModel(projection_dim=feats, recon_ps=32, decoder=decoder)
        else:
            self.enc_t1 = SimCLRModel(feats, return_z=return_z, decoder=decoder)
            self.enc_t2 = SimCLRModel(feats, return_z=return_z, decoder=decoder)

    def forward(self, t1, t2):
        return self.enc_t1(t1), self.enc_t2(t2)
    
def train_ssl_epoch(model, train_dataloader, test_dataloader, optimizer, device,
                    dense=False, temperature=0.1, lambda_rec=0.1, lambda_dense=1.0):
    """
    Adds reconstruction loss: L = L_infoNCE + lambda_rec * (L1(T1->T1) + L1(T2->T2))

    Design choices (short justification):
    - Decode SAME modality: stabilizes gradients and preserves anatomy; avoids learning a translation model.
    - Decode from h (not z): projection head is optimized for invariance/InfoNCE and tends to drop detail.
    - lambda_rec=0.1: keeps InfoNCE dominant while providing enough pressure to retain spatial detail
      (typical scale: InfoNCE ~ O(1), L1 on [0,2] patches also ~ O(0.1–1), so 0.1 is a safe start).
    """
    model.train()
    for img_t1, img_t2, _ in train_dataloader:
        img_t1 = img_t1.to(device=device, dtype=torch.float32)
        img_t2 = img_t2.to(device=device, dtype=torch.float32)

        optimizer.zero_grad()

        if dense:
            # 1. Forward pass through independent encoders
            (z_t1_global, z_t1_dense, xhat_t1) = model.enc_t1(img_t1)
            (z_t2_global, z_t2_dense, xhat_t2) = model.enc_t2(img_t2)

            # 2. Global Cross-Modal Alignment (Anatomical level)
            # We use a symmetric loss: T1->T2 and T2->T1
            loss_nce_global = (
                global_info_nce_loss(z_t1_global, z_t2_global, temperature) + 
                global_info_nce_loss(z_t2_global, z_t1_global, temperature)
            ) / 2.0

            # 3. Dense Cross-Modal Alignment (Texture/Structural level)
            # Aligning the spatial grids of T1 and T2
            loss_nce_dense = (
                dense_info_nce_loss(z_t1_dense, z_t2_dense, temperature) +
                dense_info_nce_loss(z_t2_dense, z_t1_dense, temperature)
            ) / 2.0

            # 4. Reconstruction Loss (Fidelity)
            loss_rec = F.l1_loss(xhat_t1, img_t1) + F.l1_loss(xhat_t2, img_t2)

            # 5. Total Multi-modal Loss
            # lambda_dense is usually set to 1.0 or 0.5 depending on stability
            loss = loss_nce_global + lambda_dense * loss_nce_dense + lambda_rec * loss_rec
        else:
            (z_t1, xhat_t1) = model.enc_t1(img_t1)
            (z_t2, xhat_t2) = model.enc_t2(img_t2)

            loss_nce = info_nce_multimodal([z_t1, z_t2], temperature=temperature)
            loss_rec = F.l1_loss(xhat_t1, img_t1) + F.l1_loss(xhat_t2, img_t2)

            loss = loss_nce + lambda_rec * loss_rec
        loss.backward()
        optimizer.step()

    # eval
    in_and_out = {}
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for img_t1, img_t2, _ in test_dataloader:
            img_t1 = img_t1.to(device=device, dtype=torch.float32)
            img_t2 = img_t2.to(device=device, dtype=torch.float32)

            # (z_t1, xhat_t1) = model.enc_t1(img_t1)
            # (z_t2, xhat_t2) = model.enc_t2(img_t2)

            if dense:
                # 1. Forward pass through independent encoders
                (z_t1_global, z_t1_dense, xhat_t1) = model.enc_t1(img_t1)
                (z_t2_global, z_t2_dense, xhat_t2) = model.enc_t2(img_t2)

                # 2. Global Cross-Modal Alignment (Anatomical level)
                # We use a symmetric loss: T1->T2 and T2->T1
                loss_nce_global = (
                    global_info_nce_loss(z_t1_global, z_t2_global, temperature) + 
                    global_info_nce_loss(z_t2_global, z_t1_global, temperature)
                ) / 2.0

                # 3. Dense Cross-Modal Alignment (Texture/Structural level)
                # Aligning the spatial grids of T1 and T2
                loss_nce_dense = (
                    dense_info_nce_loss(z_t1_dense, z_t2_dense, temperature) +
                    dense_info_nce_loss(z_t2_dense, z_t1_dense, temperature)
                ) / 2.0

                # 4. Reconstruction Loss (Fidelity)
                loss_rec = F.l1_loss(xhat_t1, img_t1) + F.l1_loss(xhat_t2, img_t2)

                # 5. Total Multi-modal Loss
                # lambda_dense is usually set to 1.0 or 0.5 depending on stability
                loss = loss_nce_global + lambda_dense * loss_nce_dense + lambda_rec * loss_rec
                total_loss += loss.item()
            else:
                (z_t1, xhat_t1) = model.enc_t1(img_t1)
                (z_t2, xhat_t2) = model.enc_t2(img_t2)

                loss_nce = info_nce_multimodal([z_t1, z_t2], temperature=temperature)
                loss_rec = F.l1_loss(xhat_t1, img_t1) + F.l1_loss(xhat_t2, img_t2)

                loss = loss_nce + lambda_rec * loss_rec
                total_loss += loss.item()

    in_and_out['img_t2'] = [img_t2.cpu().numpy(), xhat_t2.cpu().numpy()]

    return total_loss / len(test_dataloader), in_and_out

def test_ssl(model, dataloader, device, temperature=0.1, lambda_rec=0.1):
    model.eval()
    total_loss = 0.0
    cnt = 0
    in_and_out = {}
    with torch.no_grad():
        for batch_idx, (img_t1, img_t2, _) in enumerate(dataloader):
            img_t1 = img_t1.to(device=device, dtype=torch.float32)
            img_t2 = img_t2.to(device=device, dtype=torch.float32)
    
            (z_t1, xhat_t1) = model.enc_t1(img_t1)
            (z_t2, xhat_t2) = model.enc_t2(img_t2)

            loss_nce = info_nce_multimodal([z_t1, z_t2], temperature=temperature)
            loss_rec = F.l1_loss(xhat_t1, img_t1) + F.l1_loss(xhat_t2, img_t2)
            loss = loss_nce + lambda_rec * loss_rec

            total_loss += loss.item()
            cnt += img_t1.size(0)

        avg_loss = total_loss / cnt
        in_and_out['img_t2'] = [img_t2.cpu().numpy(), xhat_t2.cpu().numpy()]
    return avg_loss, in_and_out

if __name__ == "__main__":
    TRAIN = True
    MODE = 'dense'  # 'dense' or 'global'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if TRAIN:
        files = glob.glob("/cluster/project0/IQT_Nigeria/HCP_t1t2_ALL/sim/[!89]*/T1w/T1w_acpc_dc_restore_brain.nii.gz") #[!9]
        files_test = glob.glob("/cluster/project0/IQT_Nigeria/HCP_t1t2_ALL/sim/8*/T1w/T1w_acpc_dc_restore_brain.nii.gz")

        train_dataset = IQTDataset(
            files_t1=files,
            configs={'norm': 'zero2two', 'Data': {'mean_hr': 0.0, 'std_hr': 1.0}},
            slice_idx=(90, 160, 2),
            return_id=False,
            transform=True,
            patch=True
        )

        valid_dataset = IQTDataset(
            files_t1=files_test,
            configs={'norm': 'zero2two', 'Data': {'mean_hr': 0.0, 'std_hr': 1.0}},
            slice_idx=(100, 150, 2),
            return_id=False,
            transform=None,
            patch=True
        )

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=512,
            shuffle=True
        )

        test_dataloader = DataLoader(
            train_dataset,
            batch_size=512,
            shuffle=True,
            drop_last=False
        )

        if MODE == 'dense':
            model = MultiModalSSL(feats=128, decoder=True, dense=True)
        else:
            model = MultiModalSSL(feats=128, decoder=True, dense=False)

        optimizer = optim.Adam(
            model.parameters(),
            lr=1e-3,
            weight_decay=1e-5
        )
        model.to(device)
        best_loss = float('inf')
        early_stop = 20
        for e in range(500):
            print(f"Epoch {e+1}/10")
            avg_loss, t2_out = train_ssl_epoch(model, train_dataloader, test_dataloader, optimizer, device, temperature=0.1, lambda_rec=0.1, dense=(MODE=='dense'), lambda_dense=1.0)
            print(f"Average SSL valid loss: {avg_loss}")
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(model.state_dict(), f"best_ssl_model_ps32_recon_{MODE}_geminitransform.pth")
                print("Saved best model.")
                early_stop = 20
                # Save some output examples t2_out
                input_img, pred_img = t2_out['img_t2']
                np.savez_compressed(f"ssl_t2_recon_train_examples_{MODE}_gemini.npz", input=input_img, pred=pred_img)
            else:
                early_stop -= 1
                if early_stop == 0:
                    print("Training stopped due to early stop")
                    break
    else:
        files = glob.glob("/cluster/project0/IQT_Nigeria/HCP_t1t2_ALL/sim/9*/T1w/T1w_acpc_dc_restore_brain.nii.gz") #[!9]

        test_dataset = IQTDataset(
            files_t1=files,
            configs={'norm': 'zero2two', 'Data': {'mean_hr': 0.0, 'std_hr': 1.0}},
            slice_idx=(100, 150, 2),
            return_id=False,
            transform=False,
            patch=True,
            train=False
        )

        dataloader = DataLoader(
            test_dataset,
            batch_size=256,
            shuffle=True
        )

        model = MultiModalSSL(feats=128)
        model.load_state_dict(torch.load('/cluster/project0/IQT_Nigeria/skim/ssl_mri/best_ssl_model_patch32.pth'))
        model.to(device).float()

        avg_loss, t2_out = test_ssl(model, dataloader, device, temperature=0.1, lambda_rec=0.1)
        print(f"Test Loss: {avg_loss}")
        input_img, pred_img = t2_out['img_t2']
        np.savez_compressed("ssl_t2_recon_test_examples.npz", input=input_img, pred=pred_img)
        print("Test completed")
