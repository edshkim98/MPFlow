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

import sys
sys.path.append('/cluster/project0/IQT_Nigeria/skim/ssl_mri')
from ssl_train import SimCLRModel, MultiModalSSL

warnings.filterwarnings("ignore")

__CONDITIONING_METHOD__ = {}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

    
class ConditioningMethod(ABC):
    def __init__(self, operator, noiser, **kwargs):
        self.operator = operator
        self.noiser = noiser
        self.ssim = PatchSSIMLoss(patch_size=32)
        self.perceptual_loss = PerceptualLoss(device=device)
        self.tv = TotalVariationLoss()
        self.edge_ls = CannyEdgeLoss(low_threshold=0.05, high_threshold=0.1)
        self.ssl_model = MultiModalSSL(feats=128, return_z=False, decoder=False)
        self.ssl_model.load_state_dict(torch.load('/cluster/project0/IQT_Nigeria/skim/ssl_mri/best_ssl_model_ps32_recon.pth'), strict=False)
        self.ssl_model = self.ssl_model.to(device).float()
        self.ssl_model.eval()
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
    
    def grad_and_value(self, x_prev, x_0_hat, measurement, multimodal, t, use_lbfgs, **kwargs):
        if self.noiser.__name__ == 'gaussian':

            #x_0_hat = x_0_hat.clamp(min=0.)  ############# Ensure x_0_hat is clamped safely
            
            if use_lbfgs:
                x_0_hat = x_0_hat.detach().clone()
                x_0_hat.requires_grad_(True)
                def closure():
                    optimizer.zero_grad()
                    pred_measurement = self.operator.forward(x_0_hat, **kwargs)
                    difference = measurement - pred_measurement
                    norm = torch.linalg.norm(difference)
                    edge_ls = self.edge_ls(measurement, pred_measurement)
                    ssim = self.ssim(pred_measurement.type(torch.DoubleTensor), measurement.type(torch.DoubleTensor))
                    norm_total = norm + 0.5*edge_ls + 0.5*ssim
                    norm_total.backward()
                    return norm_total
                optimizer = LBFGS([x_0_hat], lr=0.5, max_iter=10, history_size=10, line_search_fn='strong_wolfe')
                optimizer.step(closure)
                
                # 2) Now do a new forward pass for the final gradient
                with torch.enable_grad():
                    pred_measurement = self.operator.forward(x_0_hat, **kwargs)
                    difference = measurement - pred_measurement
                    norm = torch.linalg.norm(difference)
                    edge_ls = self.edge_ls(measurement, pred_measurement)
                    ssim = self.ssim(
                        pred_measurement.type(torch.DoubleTensor),
                        measurement.type(torch.DoubleTensor)
                    )
                    new_loss = norm + 0.5*edge_ls + 0.5*ssim
                norm_grad = torch.autograd.grad(outputs=new_loss, inputs=x_prev)[0]
            else:
                pred_measurement = self.operator.forward(x_0_hat, **kwargs)
                #measurement = self.noiser(measurement)
                
                difference = measurement - pred_measurement
                norm = torch.linalg.norm(difference)
                # edge_ls = self.edge_ls(measurement, pred_measurement)
                #tv = self.tv(pred_measurement)
                # ssim = self.ssim(pred_measurement.type(torch.DoubleTensor), measurement.type(torch.DoubleTensor))
                if multimodal is not None: # and (t > 50):
                    ps = 32
                    multimodal_patches = multimodal.unfold(2, ps, ps).unfold(3, ps, ps)
                    multimodal_patches = multimodal_patches.contiguous().view(-1, 1, ps, ps)
                    x_0_hat_patches = x_0_hat.unfold(2, ps, ps).unfold(3, ps, ps)
                    x_0_hat_patches = x_0_hat_patches.contiguous().view(-1, 1, ps, ps)
                    (emb1, _) , (emb2, _) = self.ssl_model(multimodal_patches.to(device).float(), x_0_hat_patches.float())
                    ssl_ls = torch.mean((emb1 - emb2)**2)
                    #ssl_ls = torch.linalg.norm(ssl_ls)
                else:
                    ssl_ls = 0.0

                norm_total = norm + 0.5*ssl_ls
                norm_grad = torch.autograd.grad(outputs=norm_total, inputs=x_prev)[0]            
                # --- FIX: Robust Gradient Clipping ---
                # Option A: Clip by value (e.g., max change of 0.1 per step)
                grad_clip_val = 0.1
                norm_grad = torch.clamp(norm_grad, -grad_clip_val, grad_clip_val)        
        
        elif self.noiser.__name__ == 'poisson':
            Ax = self.operator.forward(x_0_hat, **kwargs)
            difference = measurement-Ax
            norm = torch.linalg.norm(difference) / measurement.abs()
            norm = norm.mean()
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]

        else:
            raise NotImplementedError
             
        return norm_grad, norm_total
   
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
        self.c1 = 1e-3      # Sufficient decrease parameter
        self.c2 = 0.8       # Curvature parameter
        self.max_line_search = 10  # Max iterations for line search
        self.alpha = self.scale_original
        self.best_ls = 1000
        self.loss_df = pd.DataFrame(columns=['Time', 'Loss'])
        self.use_lbfgs = kwargs.get('use_lbfgs', False)
        print(f"Using L-BFGS: {self.use_lbfgs}")
        
        self.csv_file = "./line_search_stepsize.csv"
        with open(self.csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Epoch", "Step Size"])
        
    def line_search(self, x_prev, x_t, x_0_hat, measurement, norm_grad, norm, t, **kwargs):
        """
        Perform line search to find step size (alpha) satisfying Wolfe conditions.
        """
        #self.alpha = self.scale_original  # Initial step size
        alpha_min = 0.3 #1e-8  # Minimum step size
        alpha_max = 1.0 #4.0*self.scale_original  # Maximum step size
        scale = self.scale

        # Original function and gradient values
        norm_orig = norm #torch.linalg.norm(measurement - self.operator.forward(x_0_hat, **kwargs))
        grad_orig = -1*norm_grad.view(-1).dot(norm_grad.view(-1))  # Norm of the gradient (directional derivative)

        for i in range(self.max_line_search):
            print(f"Line search: {i}")
            # Apply step size to get new x_t
            x_t_new = x_t - self.alpha * norm_grad
            #x_t_new.requires_grad_()  # Ensure gradient tracking #############
            
            # Compute new norm and gradient
            #with torch.no_grad():            
            if t > 0:
                x_0_hat_new = kwargs['func'](kwargs['model'], x_t_new, t-1, kwargs['clip_denoised'], kwargs['denoised_fn'], kwargs['cond_fn'], kwargs['model_kwargs'])['pred_xstart']
            
            # Ensure x_prev tracks gradients
            #x_prev = x_prev.clone().detach().requires_grad_()  #############
            
            norm_grad_new, norm_new = self.grad_and_value(x_prev=x_t_new, x_0_hat=x_0_hat_new, measurement=measurement, t=t, use_lbfgs=False, **kwargs)
            assert len(torch.unique(norm_grad_new.cpu().detach())) > 1, f"Norm grad is zero: {torch.unique(norm_grad_new.cpu().detach())}"

            # Check Wolfe conditions
            # 1. Sufficient decrease (Armijo condition)
            print("NORM New, Norm, Grad")
            print(norm_new.detach().cpu(), norm_orig + self.c1 * self.alpha * grad_orig, grad_orig)
           
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
            #grad_new = torch.abs(-1*norm_grad_new.view(-1).dot(norm_grad.view(-1)))
            print("Grad NEW, Grad_Orig")
            print(grad_new, self.c2*torch.abs(grad_orig))
            if grad_new < self.c2 * torch.abs(grad_orig):
                print("Curvature condition not met")
                self.alpha *= 1.5  # Increase step size
                if self.alpha > alpha_max:
                    self.alpha = alpha_max
                    break   # Break if maximum step size is reached
                continue
            print(f"Alpha: {self.alpha}")
            # If both conditions are satisfied, return the step size
            return self.alpha

        # If no suitable step size is found, return the minimum step size
        return self.alpha #alpha_min

    # @torch.no_grad()
    def backtracking_line_search(self, x_prev, multimodal, x0, d, y, rho, L0, g0,
                                alpha=1.0, shrink=0.5, c1=1e-4, max_ls=10, t=None, x0_anchor=False, **kwargs):
        # Armijo condition: L(x0 + a d) <= L0 + c1 * a * <g0, d>
        gd = (g0 * d).sum()
        a = alpha
        for _ in range(max_ls):
            x_try = x0 + a * d
            with torch.enable_grad():
                _, L_try = self.grad_and_value(x_prev=x_prev, x_0_hat=x_try, measurement=y, multimodal=multimodal, t=t, nonlinear_cg=x0_anchor, use_lbfgs=False, **kwargs)
            if L_try <= L0 + c1 * a * gd:
                return a
            a *= shrink
        return a  # last attempt

    def conditioning(self, x_prev, x_t, x_0_hat, measurement, multimodal, t, **kwargs):
    # Compute initial gradient and norm
        #x_prev.requires_grad_()  ###############
        #if x_t.shape[0] == 1:
        #    print("No multi-seed")
        #    self.scale = 1.0
        norm_grad, norm = self.grad_and_value(x_prev=x_prev, x_0_hat=x_0_hat, measurement=measurement, multimodal=multimodal, t=t, use_lbfgs=self.use_lbfgs, **kwargs)

        self.alpha = self.scale_original
        # Perform line search to find step size satisfying Wolfe conditions
        if (kwargs['line_search']) and (t>0):
            print(f"Time step: {t}")
            self.scale = self.line_search(x_prev.detach(), x_t.detach().requires_grad_(True), x_0_hat, measurement, norm_grad.detach(), norm, t, **kwargs)
            # print(f"Line Search Activated! Step size for iteration {t}: {self.scale}")
            # Save to CSV
            with open(self.csv_file, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([t, self.scale])
        else:
            #self.scale = self.scale_original
            if kwargs['line_search']:
                with open(self.csv_file, 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([t, self.scale])
                print(f"Step size for final iteration {t}: {self.scale}")
        
        x_t -= norm_grad * self.scale

        # Track best loss
        # if t > 0:
        #     return x_t, norm
        # print("Returning best loss: ", self.best_ls)
        # print("Re-initializing best loss")
        # self.best_ls = 1000
        #save dataframe
        self.loss_df.append({'Time': t.cpu().numpy(), 'Loss': norm.detach().cpu().item()}, ignore_index=True)
        self.loss_df.to_csv('measurement_loss_timestep.csv')
        return x_t, norm #-1*norm_grad * self.scale
        
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
