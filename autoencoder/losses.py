"""
Composite loss functions designed to produce SHARP (not blurry) reconstructions.

Three loss components:
  1. SSIM + L1  — structure-aware, no mean-regression blur
  2. VGG Perceptual — penalizes perceptual/texture differences
  3. Sobel Edge — directly penalizes blurred edges (critical for text)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from autoencoder.config import (
    LAMBDA_SSIM_L1, LAMBDA_PERCEPTUAL, LAMBDA_EDGE,
    SSIM_WEIGHT, L1_WEIGHT, DEVICE
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. SSIM Loss
# ─────────────────────────────────────────────────────────────────────────────

def _gaussian_kernel(size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """Create 2D Gaussian kernel for SSIM computation."""
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = torch.outer(g, g)
    return g / g.sum()


class SSIMLoss(nn.Module):
    """
    Structural Similarity Index loss.
    Returns (1 - SSIM) so minimizing this maximizes SSIM.
    """
    def __init__(self, window_size: int = 11, sigma: float = 1.5):
        super().__init__()
        self.window_size = window_size
        kernel = _gaussian_kernel(window_size, sigma)
        # Shape: (1, 1, window_size, window_size) → expand for 3 channels
        self.register_buffer(
            "window",
            kernel.unsqueeze(0).unsqueeze(0).expand(3, 1, -1, -1).contiguous()
        )
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Ensure window is on same device
        window = self.window.to(pred.device)
        pad = self.window_size // 2

        mu_pred = F.conv2d(pred, window, padding=pad, groups=3)
        mu_target = F.conv2d(target, window, padding=pad, groups=3)

        mu_pred_sq = mu_pred ** 2
        mu_target_sq = mu_target ** 2
        mu_cross = mu_pred * mu_target

        sigma_pred_sq = F.conv2d(pred ** 2, window, padding=pad, groups=3) - mu_pred_sq
        sigma_target_sq = F.conv2d(target ** 2, window, padding=pad, groups=3) - mu_target_sq
        sigma_cross = F.conv2d(pred * target, window, padding=pad, groups=3) - mu_cross

        ssim_map = (
            (2 * mu_cross + self.C1) * (2 * sigma_cross + self.C2)
        ) / (
            (mu_pred_sq + mu_target_sq + self.C1) * (sigma_pred_sq + sigma_target_sq + self.C2)
        )

        return 1.0 - ssim_map.mean()

    def compute_ssim_value(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Compute raw SSIM value (0 to 1, higher is better) for metrics."""
        with torch.no_grad():
            return 1.0 - self.forward(pred, target).item()


# ─────────────────────────────────────────────────────────────────────────────
# 2. VGG Perceptual Loss
# ─────────────────────────────────────────────────────────────────────────────

class VGGPerceptualLoss(nn.Module):
    """
    Perceptual loss using frozen VGG16 features.
    Compares feature maps at relu1_2, relu2_2, relu3_3 between pred and target.
    Forces the model to match textures and structures, not just pixels.
    """
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        features = vgg.features

        # Extract specific layers: relu1_2 (idx 3), relu2_2 (idx 8), relu3_3 (idx 15)
        self.slice1 = nn.Sequential(*list(features.children())[:4])   # relu1_2
        self.slice2 = nn.Sequential(*list(features.children())[4:9])  # relu2_2
        self.slice3 = nn.Sequential(*list(features.children())[9:16]) # relu3_3

        # Freeze all VGG weights — no gradient computation
        for param in self.parameters():
            param.requires_grad = False

        # ImageNet normalization (VGG expects this)
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize from [0,1] to ImageNet distribution."""
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_n = self._normalize(pred)
        target_n = self._normalize(target)

        # Extract features at each VGG level
        pred_f1 = self.slice1(pred_n)
        target_f1 = self.slice1(target_n)

        pred_f2 = self.slice2(pred_f1)
        target_f2 = self.slice2(target_f1)

        pred_f3 = self.slice3(pred_f2)
        target_f3 = self.slice3(target_f2)

        # L1 distance at each level (not MSE — avoids mean-regression)
        loss = (
            F.l1_loss(pred_f1, target_f1) +
            F.l1_loss(pred_f2, target_f2) +
            F.l1_loss(pred_f3, target_f3)
        )

        return loss


# ─────────────────────────────────────────────────────────────────────────────
# 3. Sobel Edge Loss
# ─────────────────────────────────────────────────────────────────────────────

class SobelEdgeLoss(nn.Module):
    """
    Edge-aware loss using Sobel filters.
    Applies horizontal and vertical Sobel operators to both pred and target,
    then penalizes L1 difference in edge maps.
    Directly targets blurred text edges.
    """
    def __init__(self):
        super().__init__()
        # Sobel kernels (applied per-channel)
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)

        # Shape: (1, 1, 3, 3) → expand for 3 RGB channels
        self.register_buffer(
            "sobel_x",
            sobel_x.unsqueeze(0).unsqueeze(0).expand(3, 1, -1, -1).contiguous()
        )
        self.register_buffer(
            "sobel_y",
            sobel_y.unsqueeze(0).unsqueeze(0).expand(3, 1, -1, -1).contiguous()
        )

    def _get_edges(self, x: torch.Tensor) -> torch.Tensor:
        """Compute edge magnitude using Sobel filters."""
        edges_x = F.conv2d(x, self.sobel_x.to(x.device), padding=1, groups=3)
        edges_y = F.conv2d(x, self.sobel_y.to(x.device), padding=1, groups=3)
        # Edge magnitude
        return torch.sqrt(edges_x ** 2 + edges_y ** 2 + 1e-8)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_edges = self._get_edges(pred)
        target_edges = self._get_edges(target)
        return F.l1_loss(pred_edges, target_edges)


# ─────────────────────────────────────────────────────────────────────────────
# Combined Loss
# ─────────────────────────────────────────────────────────────────────────────

class CompositeLoss(nn.Module):
    """
    Full composite loss for sharp reconstruction:
        L_total = λ1 * (α * SSIM + β * L1) + λ2 * VGG_perceptual + λ3 * Sobel_edge

    All three components fight blur in different ways:
    - SSIM: penalizes structural distortion
    - L1: pixel accuracy without mean-regression
    - VGG: perceptual texture matching
    - Sobel: direct edge sharpness enforcement
    """
    def __init__(self):
        super().__init__()
        self.ssim_loss = SSIMLoss()
        self.perceptual_loss = VGGPerceptualLoss()
        self.edge_loss = SobelEdgeLoss()
        self.l1_loss = nn.L1Loss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> dict:
        """
        Compute all loss components.
        Returns dict with individual losses and total for logging.
        """
        # 1. SSIM + L1
        ssim_val = self.ssim_loss(pred, target)
        l1_val = self.l1_loss(pred, target)
        ssim_l1 = SSIM_WEIGHT * ssim_val + L1_WEIGHT * l1_val

        # 2. VGG perceptual
        perceptual = self.perceptual_loss(pred, target)

        # 3. Sobel edge
        edge = self.edge_loss(pred, target)

        # Weighted total
        total = (
            LAMBDA_SSIM_L1 * ssim_l1 +
            LAMBDA_PERCEPTUAL * perceptual +
            LAMBDA_EDGE * edge
        )

        return {
            "total": total,
            "ssim_l1": ssim_l1.item(),
            "perceptual": perceptual.item(),
            "edge": edge.item(),
            "ssim_raw": 1.0 - ssim_val.item(),  # Actual SSIM value (higher = better)
        }


class VAELoss(nn.Module):
    """
    Composite Loss + KL Divergence for VAE.
    """
    def __init__(self, beta: float = 0.01):
        super().__init__()
        self.recon_loss = CompositeLoss()
        self.beta = beta

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> dict:
        """
        Compute reconstruction and KL divergence losses.
        """
        losses = self.recon_loss(pred, target)
        
        # KL divergence for spatial mu and logvar
        # D_KL = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Normalize KL by batch size and spatial dimensions to keep it comparable
        kl_loss = kl_loss / (mu.size(0) * mu.size(1) * mu.size(2) * mu.size(3))
        
        total_loss = losses["total"] + self.beta * kl_loss
        
        losses["kl"] = kl_loss.item()
        losses["total_vae"] = total_loss
        
        return losses


if __name__ == "__main__":
    # Quick test: verify loss computation
    loss_fn = VAELoss(beta=0.01).to(DEVICE)

    pred = torch.rand(2, 3, 512, 512, device=DEVICE)
    target = torch.rand(2, 3, 512, 512, device=DEVICE)
    mu = torch.randn(2, 512, 32, 32, device=DEVICE)
    logvar = torch.randn(2, 512, 32, 32, device=DEVICE)

    losses = loss_fn(pred, target, mu, logvar)
    for k, v in losses.items():
        val = v.item() if torch.is_tensor(v) else v
        print(f"{k}: {val:.6f}")
