from abc import ABC, abstractmethod
import torch
import torch.nn.functional as F

from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import numpy as np
import torch.nn as nn
from timm import create_model
from torch.nn.functional import interpolate
import csv
import torchvision.transforms as transforms
import pandas as pd
import warnings
from torch.optim import LBFGS
from .gaussian_diffusion import get_named_beta_schedule

warnings.filterwarnings("ignore")

__CONDITIONING_METHOD__ = {}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def extract(buf, t, x_shape):
    # t: [B] int64 indices
    out = buf.gather(0, t.long())
    return out.view(-1, *([1]*(len(x_shape)-1)))  # [B,1,1,1]

class PatchSSIMLoss(torch.nn.Module):
    def __init__(self, patch_size=32, eps=1e-6):
        """
        Initializes the Patch-based Mutual Information Loss module.
        Args:
            patch_size (int): Size of each patch (square patches are assumed).
            num_bins (int): Number of bins for the histogram.
            eps (float): Small value to avoid division by zero.
        """
        super(PatchSSIMLoss, self).__init__()
        self.patch_size = patch_size
        self.eps = eps
        self.SSIM = StructuralSimilarityIndexMeasure(data_range=1.0, win_size=11, win_sigma=1.5, K=(0.01, 0.03))

    def compute_patch_ssim(self, patch_x, patch_y):
        """
        Computes ssim for a single patch.
        Args:
            patch_x (Tensor): Patch from image X (batch_size, 1, H, W).
            patch_y (Tensor): Patch from image Y (batch_size, 1, H, W).
        Returns:
            mi (Tensor): Mutual information for the patch (scalar).
        """
        ssim_val = self.SSIM(patch_x, patch_y)
        return ssim_val

    def forward(self, x, y):
        """
        Computes the patch-based mutual information loss between two images.
        Args:
            x (Tensor): Image 1 (batch_size, 1, H, W), normalized to [0, 1].
            y (Tensor): Image 2 (batch_size, 1, H, W), normalized to [0, 1].
        Returns:
            loss (Tensor): Patch-based mutual information loss (scalar).
        """
        batch_size, _, height, width = x.size()
        ssim_loss = []
        num_patches = 0

        for i in range(0, height, self.patch_size):
            for j in range(0, width, self.patch_size):
                patch_x = x[:, :, i:i+self.patch_size, j:j+self.patch_size]
                patch_y = y[:, :, i:i+self.patch_size, j:j+self.patch_size]

                if patch_x.size(2) == self.patch_size and patch_x.size(3) == self.patch_size:
                    ssim_loss.append(1.0 - self.compute_patch_ssim(patch_x, patch_y))
                    num_patches += 1

        ssim_loss = torch.stack(ssim_loss)
        ssim_loss = torch.linalg.norm(ssim_loss)
        return ssim_loss

class CannyEdgeLoss(torch.nn.Module):
    def __init__(self, low_threshold=0.1, high_threshold=0.3):
        """
        Initializes the EdgeLoss module with thresholds suitable for normalized images.
        Args:
            low_threshold (float): Lower threshold for Canny edge detection (normalized scale 0–1).
            high_threshold (float): Higher threshold for Canny edge detection (normalized scale 0–1).
        """
        super(CannyEdgeLoss, self).__init__()
        self.low_threshold = int(low_threshold * 255)  # Scale for OpenCV (expects 0-255)
        self.high_threshold = int(high_threshold * 255)  # Scale for OpenCV (expects 0-255)

        # Example 3x3 Sobel kernels:
        self.sobel_x = torch.tensor([[-1., 0., 1.],
                                [-2., 0., 2.],
                                [-1., 0., 1.]], dtype=torch.float32).reshape(1,1,3,3).to(device)
        self.sobel_y = torch.tensor([[-1., -2., -1.],
                                [ 0.,  0.,  0.],
                                [ 1.,  2.,  1.]], dtype=torch.float32).reshape(1,1,3,3).to(device)

    def sobel_edge_magnitude(self,image):
        """Compute a differentiable approximation of edge magnitude via Sobel."""
        #print data type
        grad_x = F.conv2d(image, self.sobel_x, padding=1)
        grad_y = F.conv2d(image, self.sobel_y, padding=1)
        # Edge magnitude
        edges = torch.sqrt(grad_x**2 + grad_y**2 + 1e-7)
        return edges

    def forward(self, image_A, image_B):
        """
        Compute edge loss between two normalized images.
        Input:
            image_A: PyTorch tensor (batch_size, C, H, W), normalized to [0, 1].
            image_B: PyTorch tensor (batch_size, C, H, W), normalized to [0, 1].
        Output:
            loss: Scalar edge loss.
        """
        
        # Check if the input images are normalized to [0, 1]
        if image_A.max() > 1:
            image_A = image_A / 2.0
        if image_B.max() > 1:
            image_B = image_B / 2.0

        edges_A = self.sobel_edge_magnitude(image_A.to(torch.float32))
        edges_B = self.sobel_edge_magnitude(image_B.to(torch.float32))
        difference = edges_A - edges_B
        loss = torch.linalg.norm(difference)  # sum of all differences
        return loss

class TotalVariationLoss(nn.Module):
    def __init__(self):
        """
        Initialize the Total Variation Loss module.
        """
        super(TotalVariationLoss, self).__init__()

    def forward(self, image):
        """
        Compute the Total Variation (TV) Loss for an image.
        
        Args:
            image (torch.Tensor): Input image of shape (B, C, H, W),
                                  where B is batch size, C is the number of channels,
                                  H is the height, and W is the width.
        
        Returns:
            tv_loss (torch.Tensor): Scalar tensor representing the total variation loss.
        """
        # Compute horizontal and vertical differences
        diff_h = torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :])  # Horizontal differences
        diff_w = torch.abs(image[:, :, :, 1:] - image[:, :, :, :-1])  # Vertical differences
        
        # Sum the absolute differences
        tv_loss = diff_h.sum() + diff_w.sum()
        
        return tv_loss
    
class PerceptualLoss(nn.Module):
    def __init__(self, model_name="resnet18", layers=("layer2", "layer3"), device='cuda'):
        """
        Perceptual Loss using intermediate features of a Timm pre-trained model.
        Args:
            model_name: Name of the model to use from Timm (e.g., "resnet18").
            layers: Tuple of layer names from which to extract intermediate features.
        """
        super(PerceptualLoss, self).__init__()
        self.layers = layers
        self.device = device

        # Load a pre-trained model from Timm
        self.feature_extractor = create_model(model_name, pretrained=True, features_only=True, out_indices=tuple(range(len(layers))))
        self.layer_names = [f"layer{i}" for i in range(len(self.feature_extractor.feature_info))]

        # Freeze the feature extractor's parameters
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.feature_extractor.to(self.device)

    def forward(self, input_image, target_image):
        """
        Calculate perceptual loss between input and target images.
        Args:
            input_image: Tensor of shape (B, 1, H, W), normalized to [0, 1].
            target_image: Tensor of shape (B, 1, H, W), normalized to [0, 1].
        Returns:
            loss: Scalar perceptual loss.
        """
        # Convert grayscale (1-channel) images to 3-channel by repeating
        input_image = input_image.repeat(1, 3, 1, 1)  # Shape: (B, 3, H, W)
        target_image = target_image.repeat(1, 3, 1, 1)  # Shape: (B, 3, H, W)
        
        # Normalize images using ImageNet statistics
        #input_image = (input_image - 0.485) / 0.229
        #target_image = (target_image - 0.485) / 0.229

        # Ensure input and target are resized to match the model's expected input size
        input_image = interpolate(input_image, size=(256, 256), mode="bilinear", align_corners=False)
        target_image = interpolate(target_image, size=(256, 256), mode="bilinear", align_corners=False)

        # Extract intermediate features
        input_features = self.feature_extractor(input_image)
        target_features = self.feature_extractor(target_image)

        # Calculate perceptual loss using L2 norm of feature differences
        loss = 0.0
        for input_feat, target_feat in zip(input_features, target_features):
            loss += torch.linalg.norm(input_feat - target_feat)

        return loss


def register_conditioning_method(name: str):
    def wrapper(cls):
        if __CONDITIONING_METHOD__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __CONDITIONING_METHOD__[name] = cls
        return cls
    return wrapper

def get_conditioning_method(name: str, operator, noiser, **kwargs):
    if __CONDITIONING_METHOD__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined!")
    return __CONDITIONING_METHOD__[name](operator=operator, noiser=noiser, **kwargs)


@torch.no_grad()
def bb_stepsize_torch(xk, xkm1, gk, gkm1, kind="BB1",
                      alpha_min=1e-8, alpha_max=1e3, fallback=None):
    """
    Barzilai–Borwein step size per batch item.

    Args:
      xk, xkm1: current and previous iterates, shape [B, C, H, W] (or any)
      gk, gkm1: current and previous gradients, same shape
      kind: "BB1" (aggressive) or "BB2" (conservative)
      alpha_min/max: clamp range
      fallback: optional scalar to use when denominator <= 0 or tiny (default: 1.0)

    Returns:
      alpha: shape [B] (per-sample scalar step sizes)
    """
    B = xk.shape[0]
    s = (xk - xkm1).reshape(B, -1)          # s_k = x_k - x_{k-1}
    y = (gk - gkm1).reshape(B, -1)          # y_k = g_k - g_{k-1}

    sty   = (s * y).sum(dim=1)              # s^T y
    sTs   = (s * s).sum(dim=1)              # s^T s
    yTy   = (y * y).sum(dim=1)              # y^T y

    if kind.upper() == "BB1":
        numer, denom = sTs, sty             # α = (s^T s) / (s^T y)
    else:  # BB2
        numer, denom = sty, yTy             # α = (s^T y) / (y^T y)

    # handle non-convex / noisy cases
    if fallback is None:
        fallback = 1.0

    # where denom <= 0 or tiny, fall back
    bad = (denom <= 1e-16) | torch.isnan(denom) | torch.isinf(denom)
    alpha = torch.empty_like(denom)
    alpha[bad]  = fallback
    alpha[~bad] = numer[~bad] / denom[~bad]
    #alpha = numer / denom
    print("ALPHA Real: ", numer / denom)

    # clamp
    alpha = torch.clip(alpha, alpha_min, alpha_max)
    return alpha

class DC_BB_State:
    def __init__(self):
        self.x0_prev = None
        self.g_prev  = None
        self.k       = 0         # iteration counter
        self.alpha_last = None   # remembered step for fallback

    def reset(self):
        self.x0_prev = None
        self.g_prev = None
        self.k = 0
        self.alpha_last = None

class CurvatureEMA:
    """
    Keeps an EMA of the absolute diagonal Hessian estimate
    and returns a per‑pixel step‑size map α_t.
    """
    def __init__(self, shape, decay=0.9,
                 eps_reg=1e-4,  # numerical stabiliser
                 alpha_min=1.0, alpha_max=3.0):
        self.decay      = decay
        self.eps_reg    = eps_reg
        self.a_min      = alpha_min
        self.a_max      = alpha_max
        self.buffer     = torch.zeros(shape, device='cuda')  # init to zero

    @torch.no_grad()
    def update_and_get_alpha(self, diag_est):
        """
        diag_est : (B,C,H,W) Hutchinson diagonal for the *current* step
        Returns   : α_map  (same shape)   ready to scale the DC gradient
        """
        curv_now = diag_est.abs()                 # magnitude only
        self.buffer.mul_(self.decay).add_((1. - self.decay) * curv_now)

        # inverse‑√‑variance scaling
        alpha_map = 1.0 / (self.eps_reg + self.buffer)
        alpha_map.clamp_(self.a_min, self.a_max)
        return alpha_map
    
class ConditioningMethod(ABC):
    def __init__(self, operator, noiser, **kwargs):
        self.operator = operator
        self.noiser = noiser
        self.ssim = PatchSSIMLoss(patch_size=32)
        self.perceptual_loss = PerceptualLoss(device=device)
        self.tv = TotalVariationLoss()
        self.edge_ls = CannyEdgeLoss(low_threshold=0.05, high_threshold=0.1)
        self.cnt = 0
        self.ema_trace = CurvatureEMA(shape=(1, 1, 256, 256), decay=0.9)
        self.alpha_df = pd.DataFrame(columns=['Time', 'StepMin', 'StepMax', 'StepMean', 'StepVar'])

        # build buffers once
        self.betas = torch.tensor(get_named_beta_schedule("cosine", 1000), dtype=torch.float32).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0).to(device)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod).to(device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod).to(device)
        self.reduced_alpha_cumprod = torch.divide(self.alphas_cumprod, self.alphas_cumprod).to(device)
        #print(f"Every 100 alphas: {self.alphas[::100]}")
        self.alpha_map_lst = []
        self.cnt = 0


    def edge_loss(self, image1, image2):
        """
        Compute edge loss between two 1-channel images using Sobel filters.
        The loss is the mean squared error (MSE) between the edge maps of the two images.

        Args:
            image1 (torch.Tensor): First image, shape (B, 1, H, W), values in [0, 1].
            image2 (torch.Tensor): Second image, shape (B, 1, H, W), values in [0, 1].

        Returns:
            torch.Tensor: Scalar edge loss.
        """

        def sobel_filter(image):
            # Sobel filters for edge detection
            sobel_x = torch.tensor([[-1, 0, 1],
                                    [-2, 0, 2],
                                    [-1, 0, 1]], device=image.device).view(1, 1, 3, 3)
            sobel_x = sobel_x.to(image.dtype)
            sobel_y = torch.tensor([[-1, -2, -1],
                                    [ 0,  0,  0],
                                    [ 1,  2,  1]], device=image.device).view(1, 1, 3, 3)
            sobel_y = sobel_y.to(image.dtype)
            # Apply Sobel filters in x and y directions
            grad_x = F.conv2d(image, sobel_x, padding=1)  # Gradient in x-direction
            grad_y = F.conv2d(image, sobel_y, padding=1)  # Gradient in y-direction
            
            # Compute gradient magnitude
            grad_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)
            return grad_magnitude

        # Compute edge maps for both images
        edge_map1 = sobel_filter(image1.to(torch.float32))
        edge_map2 = sobel_filter(image2.to(torch.float32))

        # Compute mean squared error between edge maps
        difference = edge_map1 - edge_map2
        loss = torch.linalg.norm(difference)
        #loss = F.mse_loss(edge_map1, edge_map2)
        return loss    

    def project(self, data, noisy_measurement, **kwargs):
        return self.operator.project(data=data, measurement=noisy_measurement, **kwargs)

    def grad_and_value(self, x_prev, x_0_hat, measurement, t, use_secondtweedie, score, spatial_weight, diffpir, daps, reddiff, **kwargs):

        if self.noiser.__name__ == 'gaussian':
            
            if not reddiff:
                x_0_hat = x_0_hat.clamp(min=0., max=2.)  ############# Ensure x_0_hat is clamped safely
            #if t <= 0:
            self.alpha_map = torch.tensor(1.0)

            if (use_secondtweedie==True) and (t>0):
                pred_measurement = self.operator.forward(x_0_hat, **kwargs)
                difference = measurement - pred_measurement
                #norm = torch.linalg.norm(difference)

                def trace_estimate(x_prev, score, eps=None, **kwargs):
                    """
                    Hutchinson estimator of tr ∇² log p(z) using one noise vector.
                    z          : latent tensor, requires_grad=True
                    score_fn   : function returning ∇ log p(z)  (same shape as z)
                    """
                    alpha_map = torch.tensor(1.0)
                    s = score
                    diag_sum = 0
                    k = 3

                    sigma_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_prev.shape)
                    h_sigma = 0.25 * sigma_t

                    data_range = 2.0  # since values are in [0, 2]; or compute per-batch from x_prev
                    h_min = (4.0/255.0) * data_range   # ≈ 4/255 for [0,2]  (~0.0157)
                    h_max = 0.10 * data_range          # ≈ 0.2  for [0,2]
                    h = torch.clamp(h_sigma, min=h_min, max=h_max)

                    for _ in range(k):
                        #if eps is None:
                        eps = torch.randn_like(x_prev)
                        eps = eps / (eps.abs().amax(dim=(1,2,3), keepdim=True) + 1e-12)
                        #Rademacher noise
                        #eps = torch.sign(eps)

                    #s  = score #score_fn(z)           # ∇ log p(z)
                        s_ = kwargs['func'](kwargs['model'], (x_prev + h*eps), t, kwargs['clip_denoised'], kwargs['denoised_fn'], kwargs['cond_fn'], kwargs['model_kwargs'])['score'] # ∇ log p(z + ε)
                    #s_ = score_fn(z + eps)     # ∇ log p(z + ε)

                        if spatial_weight:
                            diag_sum += (eps * ((s_ - s) / h))
                        else:
                            # scalar trace estimate per sample in the batch
                            diag_sum += (eps * ((s_ - s) / h))
                            #diag_est = diag_est.clamp(None, 0.0)
                    
                    diag_sum /= k
                    #diag_sum = diag_sum.clamp(None, 0.0)
                    diag_pool = F.avg_pool2d(diag_sum.abs(), 3, 1, 1) #abs()
                    alpha_map  = (1.0 / (1e-4 + diag_pool))
                    alpha_map = alpha_map.clamp(0.5, 2.0)
                    #alpha_map /= alpha_map.mean()
                    tr_est = diag_sum.view(x_prev.shape[0], -1).sum(dim=1).mean()
                    #if t == 199:
                    #    np.save(f"/cluster/project0/IQT_Nigeria/skim/diffusion_inverse/guided-diffusion/alpha_map_t199.npy", alpha_map.detach().cpu().numpy())
                    self.alpha_df = self.alpha_df.append({'Time': t.cpu().numpy(), 'StepMin': alpha_map.detach().cpu().numpy().min(), 'StepMax': alpha_map.detach().cpu().numpy().max(), 'StepMean': alpha_map.detach().cpu().numpy().mean(), 'StepVar': alpha_map.detach().cpu().numpy().var()}, ignore_index=True)
                    self.alpha_df.to_csv('alpha_map_dps_spatial.csv')
                    # if t < 140 and t > 110:
                    #     self.alpha_map_lst.append(alpha_map.detach().cpu().numpy())
                    # elif t == 110:
                    #     self.alpha_map_lst = np.mean(np.concatenate(self.alpha_map_lst, axis=0), axis=0)
                    #     np.save(f"alpha_map_mean_{self.cnt}.npy", self.alpha_map_lst)
                    #     self.cnt += 1
                    #     self.alpha_map_lst = []

                    return tr_est, alpha_map
                
                # 2) Now do a new forward pass for the final gradient
                with torch.enable_grad():
                    # ------------ usage -------------
                    eta = 0.02                             # paper's hyper-parameter
                    d   = x_prev[0].numel()                    # latent dimensionality

                    trace, alpha_map = trace_estimate(x_prev= x_prev, score=score, **kwargs)   # forward pass (2× score network)
                    loss_trace  =  trace             # part of surrogate loss
                    
                    norm = torch.linalg.norm(difference)
                    total_loss = norm #+ loss_trace

                if diffpir:
                    norm_grad = torch.autograd.grad(outputs=total_loss, inputs=x_0_hat, retain_graph=False)[0]
                else:
                    norm_grad = torch.autograd.grad(outputs=total_loss, inputs=x_prev, retain_graph=False)[0]
                trace_grad = None #torch.autograd.grad(outputs=(1.0/loss_trace), inputs=x_prev)[0]
                norm_grad *= (1.0 / alpha_map.detach())
                #trace_grad *= (eta / d)
                norm_grad = [norm_grad, trace_grad]
                #norm_grad += trace_grad

                self.alpha_map = alpha_map.detach()
                
            else:
                pred_measurement = self.operator.forward(x_0_hat, **kwargs)
                #measurement = self.noiser(measurement)
                
                difference = measurement - pred_measurement
                norm = torch.linalg.norm(difference)
                #edge_ls = self.edge_ls(measurement, pred_measurement)
                #tv = self.tv(pred_measurement)
                #ssim = self.ssim(pred_measurement.type(torch.DoubleTensor), measurement.type(torch.DoubleTensor))
                # pred_measurement[pred_measurement < 0.] = 0.
                # measurement[measurement < 0.] = 0.
                #percept = self.perceptual_loss(pred_measurement.type(torch.float32), measurement.type(torch.float32))
                norm_total = norm #+ 0.5*edge_ls + 0.1*ssim #+ 0.005*tv
                if diffpir:
                    norm_grad = torch.autograd.grad(outputs=norm_total, inputs=x_0_hat)[0]
                elif daps:
                    pred_measurement = self.operator.forward(x_prev, **kwargs)
                    difference = measurement - pred_measurement
                    norm = torch.linalg.norm(difference)
                    #norm = 0.5 * (difference ** 2).mean() 
                    norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0] #Here x_prev is x_t
                elif reddiff:

                    x0_pred = x_0_hat + kwargs["sigma_x0"] * kwargs['noise_x0']        # noise_x0 ~ N(0, I); NO detach on mu; noise can be detached, x0_hat is mu
                    x0_pred = torch.clamp(x0_pred, 0.0, 2.0)
                    pred_measurement = self.operator.forward(x0_pred, **kwargs) #x_prev is pred_x0

                    difference = measurement - pred_measurement
                    #norm = torch.linalg.norm(difference)
                    norm = (difference ** 2).mean()

                    loss_noise = torch.mul(kwargs["residual_noise"], x0_pred)
                    #Mean loss noise so its scalar
                    loss_noise = loss_noise.mean()

                    norm_total = kwargs['d_t'] * norm + kwargs['w_t'] * loss_noise
                    norm_grad = norm_total
                    #norm_grad = torch.autograd.grad(outputs=norm_total, inputs=x_0_hat)[0] # x_0_hat = mu
                else:
                    norm_grad = torch.autograd.grad(outputs=norm_total, inputs=x_prev)[0]
                norm_grad *= self.alpha_map
                #if t < 140 and t > 110:
                #    self.alpha_map_lst.append(norm_grad.detach().cpu().numpy())
                #elif t == 110:
                #    self.alpha_map_lst = np.mean(np.concatenate(self.alpha_map_lst, axis=0), axis=0)
                #    np.save(f"grad_mean_{self.cnt}.npy", self.alpha_map_lst)
                #    self.cnt += 1
                #    self.alpha_map_lst = []

        
        elif self.noiser.__name__ == 'poisson':
            Ax = self.operator.forward(x_0_hat, **kwargs)
            difference = measurement-Ax
            norm = torch.linalg.norm(difference) / measurement.abs()
            norm = norm.mean()
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]

        else:
            raise NotImplementedError
             
        return norm_grad, norm
   
    @abstractmethod
    def conditioning(self, x_t, measurement, noisy_measurement=None, **kwargs):
        pass
    
@register_conditioning_method(name='vanilla')
class Identity(ConditioningMethod):
    # just pass the input without conditioning
    def conditioning(self, x_t):
        return x_t
    
@register_conditioning_method(name='projection')
class Projection(ConditioningMethod):
    def conditioning(self, x_t, noisy_measurement, **kwargs):
        x_t = self.project(data=x_t, noisy_measurement=noisy_measurement)
        return x_t


@register_conditioning_method(name='mcg')
class ManifoldConstraintGradient(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.scale = kwargs.get('scale', 1.0)
        
    def conditioning(self, x_prev, x_t, x_0_hat, measurement, noisy_measurement, **kwargs):
        # posterior sampling
        norm_grad, norm = self.grad_and_value(x_prev=x_prev, x_0_hat=x_0_hat, measurement=measurement, **kwargs)
        x_t -= norm_grad * self.scale
        
        # projection
        x_t = self.project(data=x_t, noisy_measurement=noisy_measurement, **kwargs)
        return x_t, norm
        
@register_conditioning_method(name='ps')
class PosteriorSampling(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.scale = kwargs.get('scale', 1.0)
        self.scale_original = self.scale
        self.c1 = 1e-4 #5e-3      # Sufficient decrease parameter
        self.c2 = 0.05 #0.95       # Curvature parameter
        self.max_line_search = 10  # Max iterations for line search
        self.alpha = self.scale_original
        self.best_ls = 1000
        self.loss_df = pd.DataFrame(columns=['Time', 'Loss'])
        self.use_secondtweedie = kwargs.get('use_secondtweedie', False)
        self.diffpir = kwargs.get('diffpir', False)
        self.daps = kwargs.get('daps', False)
        self.reddiff = kwargs.get('reddiff', False)
        if self.diffpir:
            self.zeta = 0.1
        self.bb_linesearch = kwargs.get('BB_line_search', False)
        if self.daps:
            self.step_size = self.scale       # η
            self.noise_scale = kwargs.get('noise_scale', 1.0)       # multiplier on sqrt(2η)ξ
            self.likelihood_weight = kwargs.get('likelihood_weight', 1.0)

        if self.bb_linesearch:
            self.state = DC_BB_State()
            self.alt_bb = True
        print(f"Using Second Order: {self.use_secondtweedie}")

        
        self.csv_file = f"./line_search_stepsize_dynamicdps_full_{self.c2}.csv"
        with open(self.csv_file, "w", newline="") as f:
           writer = csv.writer(f)
           writer.writerow(["Epoch", "Step Size"])
       
    def line_search(self, x_prev, x_t, x_0_hat, measurement, norm_grad, norm, t, use_secondtweedie, score, spatial_weight, **kwargs):
        """
        Perform line search to find step size (alpha) satisfying Wolfe conditions.
        """
        #self.alpha = self.scale_original  # Initial step size
        alpha_min = 0.05 #1e-8  # Minimum step size
        alpha_max = 2.0 #4.0*self.scale_original  # Maximum step size
        scale = self.scale

        # Original function and gradient values
        norm_orig = norm #torch.linalg.norm(measurement - self.operator.forward(x_0_hat, **kwargs))
        grad_orig = -1*norm_grad.view(-1).dot(norm_grad.view(-1))  # Norm of the gradient (directional derivative)

        for i in range(self.max_line_search):
            print(f"Line search: {i}")
            # Apply step size to get new x_t
            x_t_new = (x_t - self.alpha * norm_grad)
            #x_t_new = x_t_new.detach().requires_grad_(True)  # Ensure gradient tracking #############
           
            #print("x_t_new.requires_grad:", x_t_new.requires_grad)    # MUST be True
            #print("x_t_new.grad_fn:", x_t_new.grad_fn)                # MUST not be None
 
            # Compute new norm and gradient
            #with torch.no_grad():            
            #if t > 0:
            x_0_hat_new = kwargs['func'](kwargs['model'], x_t_new, t-1, kwargs['clip_denoised'], kwargs['denoised_fn'], kwargs['cond_fn'], kwargs['model_kwargs'])['pred_xstart'] #Clip_denoised is set False

            #print("x_0_hat_new.requires_grad:", x_0_hat_new.requires_grad)    # MUST be True
            #print("x_0_hat_new.grad_fn:", x_0_hat_new.grad_fn)                # MUST not be None   
             
            norm_grad_new, norm_new = self.grad_and_value(x_prev=x_t_new, x_0_hat=x_0_hat_new, measurement=measurement, t=t, use_secondtweedie=use_secondtweedie, score=score, spatial_weight=spatial_weight, diffpir=self.diffpir, daps=self.daps, reddiff=self.reddiff, **kwargs)
            #assert len(torch.unique(norm_grad_new.cpu().detach())) > 1, f"Norm grad is zero: {torch.unique(norm_grad_new.cpu().detach())} Norm: {norm_new.cpu()} X_hat: {torch.unique(x_0_hat_new.cpu())}"

            # Check Wolfe conditions
            # 1. Sufficient decrease (Armijo condition)
            #print("NORM New, Norm, Grad")
            #print(norm_new.detach().cpu(), norm_orig + self.c1 * self.alpha * grad_orig, grad_orig)
           
            if norm_new > norm_orig + self.c1 * self.alpha * grad_orig:
                print("Armjiho condition not met")
                self.alpha *= 0.75  # Reduce step size
                # print(norm_new, norm_orig + self.c1 * alpha * grad_orig)
                if self.alpha < alpha_min:
                    self.alpha = alpha_min
                    break   # Break if minimum step size is reached
                continue

            # 2. Curvature condition
            grad_new = torch.abs(-1*norm_grad_new.view(-1).dot(norm_grad.view(-1)))
            #print(grad_new, grad_orig)
            if grad_new > self.c2 * torch.abs(grad_orig): ############### This has been changed originak: <
                print("Curvature condition not met")
                self.alpha *= 1.25  # Increase step size
                if self.alpha > alpha_max:
                    self.alpha = alpha_max
                    break   # Break if maximum step size is reached
                continue

            # If both conditions are satisfied, return the step size
            return self.alpha
        # If no suitable step size is found, return the minimum step size
        return self.alpha #alpha_min
    
    def conditioning(self, x_prev, x_t, x_0_hat, score, measurement, t, **kwargs):

        if self.reddiff:
            norm_grad, norm = self.grad_and_value(x_prev=x_t, x_0_hat=x_0_hat, measurement=measurement, t=t, use_secondtweedie=self.use_secondtweedie, score=score, spatial_weight=False, diffpir=False, daps=False, reddiff=True, **kwargs)
            #write w_k step size
            #with open(self.csv_file, 'a', newline='') as file:
            #   writer = csv.writer(file)
            #   writer.writerow([t, kwargs['d_t']*self.scale])
            
            #if norm.item() < 0.65:
            #    np.save('pred_x0.npy', x_prev.detach().cpu().numpy())
            #    np.save('mu.npy', x_0_hat.detach().cpu().numpy())

            #mu = x_0_hat 
            #mu -= self.scale * norm_grad.detach()
            self.loss_df = self.loss_df.append({'Time': t.cpu().numpy(), 'Loss': norm.detach().cpu().numpy()}, ignore_index=True)
            self.loss_df.to_csv('measurement_loss_timestep_reddiff_vanilla.csv')

            return norm_grad, norm

        if self.daps:
            # Just return the output of self.grad_and_value
            x_t = x_t.detach().requires_grad_(True)
            norm_grad, norm = self.grad_and_value(x_prev=x_t, x_0_hat=x_0_hat, measurement=measurement, t=t, use_secondtweedie=self.use_secondtweedie, score=score, spatial_weight=False, diffpir=False, daps=True, reddiff=False, **kwargs)
            self.loss_df = self.loss_df.append({'Time': t.cpu().numpy(), 'Loss': norm.detach().cpu().numpy()}, ignore_index=True)
            self.loss_df.to_csv('measurement_loss_timestep_daps_vanilla_geometric.csv')
            return norm_grad, norm

        if self.bb_linesearch:
            self.scale = self.scale_original

        if self.diffpir:
            sigma = 0.001
            #snr_sqrt = self.reduced_alpha_cumprod[t]
            # sigma_ks = self.sqrt_one_minus_alphas_cumprod[t]/self.sqrt_alphas_cumprod[t]
            # if t < 300:
            #     self.scale = 2.0
            # if t < 100:
            #     self.scale = 1.0
            # if t < 10:
            #     self.scale = 0.1
            rho_t = self.scale #self.scale*(sigma**2)/(sigma_ks**2)
            # print(f"Sigma_k: {sigma_ks}, Rho_t: {rho_t}")

    # Compute initial gradient and norm
        #x_prev.requires_grad_()  ###############
        norm_grad, norm = self.grad_and_value(x_prev=x_prev, x_0_hat=x_0_hat, measurement=measurement, t=t, use_secondtweedie=self.use_secondtweedie, score=score, spatial_weight=True, diffpir=self.diffpir, daps=self.daps, reddiff=self.reddiff,**kwargs)
        if self.use_secondtweedie:
            try:
                norm_grad, trace_grad = norm_grad[0], norm_grad[1]
            except:
                trace_grad = 0.
        #Add t and norm to dataframe
        self.loss_df = self.loss_df.append({'Time': t.cpu().numpy(), 'Loss': norm.detach().cpu().numpy()}, ignore_index=True)
        #print(f"Loss at {t.cpu().detach().numpy()}: {norm}")
        if self.best_ls > norm:
            self.best_ls = norm
            self.best_x = x_0_hat

        self.alpha = self.scale_original
        # Perform line search to find step size satisfying Wolfe conditions
        if (kwargs['line_search']) and (t>0):
            self.scale = self.line_search(x_prev.detach(), x_t.detach().requires_grad_(True), x_0_hat, measurement, norm_grad.detach(), norm, t, use_secondtweedie=False, score=None, spatial_weight=False, **kwargs)
            rho_t = self.scale
            # Save to CSV
            with open(self.csv_file, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([t, self.scale])

        else:
            if kwargs['line_search']:
                print(f"Step size for final iteration {t}: {self.scale}")

        if self.diffpir:
            x_0_hat -= norm_grad * rho_t
            #with open(self.csv_file, 'a', newline='') as file:
             #   writer = csv.writer(file)
             #   writer.writerow([t, rho_t])

            #t_im1 = utils_model.find_nearest(self.reduced_alpha_cumprod, snr_sqrt.cpu().numpy()) only true if time steps are not 1000
            #t_im1 = t
            #eps = (x_t - self.sqrt_alphas_cumprod[t] * x_0_hat) / self.sqrt_one_minus_alphas_cumprod[t]
            # calculate \hat{\eposilon}
            #eta_sigma = self.sqrt_one_minus_alphas_cumprod[t_im1] / self.sqrt_one_minus_alphas_cumprod[t] * torch.sqrt(self.betas[t])
            #x_t = self.sqrt_alphas_cumprod[t_im1] * x_0_hat + np.sqrt(1-self.zeta) * (torch.sqrt(self.sqrt_one_minus_alphas_cumprod[t_im1]**2 - eta_sigma**2) * eps \
            #            + eta_sigma * torch.randn_like(x_t)) + np.sqrt(self.zeta) * self.sqrt_one_minus_alphas_cumprod[t_im1] * torch.randn_like(x_t)

            # --- 4) form x_{t-1} directly from updated x0_new (x0-form) ---
            sqrt_ac_tm1 = torch.sqrt(self.alphas_cumprod.gather(0, t)).view(-1,1,1,1)
            sqrt_1m_tm1 = self.sqrt_one_minus_alphas_cumprod.gather(0, t).view(-1,1,1,1)

            # # stochastic DDPM step (set noise=0 for deterministic DDIM-like)
            noise = torch.randn_like(x_t)
            x_t = sqrt_ac_tm1 * x_0_hat + sqrt_1m_tm1 * noise

        elif self.use_secondtweedie and not self.diffpir:
            x_t -= norm_grad * self.scale #- trace_grad
        else:
            x_t -= norm_grad * self.scale
        #diff = measurement - x_0_hat
        #extrinsic_loss = torch.linalg.norm(diff)
        #x_t += extrinsic_loss * self.scale * 0.5
        #if (t % 10 == 0) and (t <= 100):
        #    np.save(f"x_pred_{t.detach().cpu().numpy()}.npy", x_0_hat.detach().cpu().numpy())
        if t > 0:
            return x_t, norm
        print("Returning best loss: ", self.best_ls)
        print("Re-initializing best loss")
        self.best_ls = 1000
        #save dataframe
        self.loss_df.to_csv(f'measurement_loss_timestep_dynamicdps_vanilla_full_{self.c2}.csv')
        return x_t, norm #self.best_x, self.best_ls
        
@register_conditioning_method(name='ps+')
class PosteriorSamplingPlus(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.num_sampling = kwargs.get('num_sampling', 5)
        self.scale = kwargs.get('scale', 1.0)

    def conditioning(self, x_prev, x_t, x_0_hat, measurement, **kwargs):
        norm = 0
        for _ in range(self.num_sampling):
            # TODO: use noiser?
            x_0_hat_noise = x_0_hat + 0.05 * torch.rand_like(x_0_hat)
            difference = measurement - self.operator.forward(x_0_hat_noise)
            norm += torch.linalg.norm(difference) / self.num_sampling
        
        norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]
        x_t -= norm_grad * self.scale
        return x_t, norm
