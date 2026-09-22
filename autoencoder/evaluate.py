"""
Evaluation script for the trained autoencoder.

Loads the best checkpoint, runs on the validation set, and produces:
  1. Per-image SSIM and PSNR metrics (mean ± std)
  2. Side-by-side comparison grids (original | reconstructed)
  3. Zoomed-in text region crops for sharpness verification
  4. Summary metrics table
"""

import os
import torch
import numpy as np
from PIL import Image
import torchvision.utils as vutils
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm

from autoencoder.config import DEVICE, CHECKPOINT_DIR, OUTPUT_DIR
from autoencoder.model import DocumentAutoencoder
from autoencoder.losses import SSIMLoss
from autoencoder.dataset import get_dataloaders


def load_best_model() -> DocumentAutoencoder:
    """Load the best checkpoint."""
    ckpt_path = os.path.join(CHECKPOINT_DIR, "best_autoencoder.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"No checkpoint found at {ckpt_path}. Train the model first with:\n"
            f"  python -m autoencoder.train"
        )

    model = DocumentAutoencoder().to(DEVICE)
    checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"[Eval] Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"[Eval] Best val SSIM during training: {checkpoint.get('best_ssim', 0.0):.4f}")

    return model, checkpoint


def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute PSNR in dB for a single image pair."""
    mse = torch.mean((pred - target) ** 2).item()
    if mse < 1e-10:
        return 100.0
    return 10 * np.log10(1.0 / mse)


@torch.no_grad()
def evaluate_metrics(model, val_loader, device):
    """Compute per-image SSIM and PSNR on the full validation set."""
    ssim_fn = SSIMLoss().to(device)
    ssim_scores = []
    psnr_scores = []

    print("[Eval] Computing metrics on validation set...")
    for inputs, targets in tqdm(val_loader, desc="Evaluating"):
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs)

        # Per-image metrics
        for i in range(inputs.size(0)):
            pred_i = outputs[i:i+1]
            target_i = targets[i:i+1]

            ssim_val = ssim_fn.compute_ssim_value(pred_i, target_i)
            psnr_val = compute_psnr(pred_i, target_i)

            ssim_scores.append(ssim_val)
            psnr_scores.append(psnr_val)

    ssim_arr = np.array(ssim_scores)
    psnr_arr = np.array(psnr_scores)

    return ssim_arr, psnr_arr


@torch.no_grad()
def save_comparison_results(model, val_loader, device, num_samples=8):
    """Save detailed comparison images."""
    originals = []
    reconstructed = []

    for inputs, targets in val_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs)

        originals.append(targets.cpu())
        reconstructed.append(outputs.cpu())

        if sum(o.size(0) for o in originals) >= num_samples:
            break

    originals = torch.cat(originals, dim=0)[:num_samples]
    reconstructed = torch.cat(reconstructed, dim=0)[:num_samples]

    # ─── Full comparison grid ────────────────────────────────────────────
    comparison = torch.stack([originals, reconstructed], dim=1)
    comparison = comparison.view(-1, *originals.shape[1:])
    grid = vutils.make_grid(comparison, nrow=2, padding=4, pad_value=0.5)
    grid_path = os.path.join(OUTPUT_DIR, "eval_comparison_grid.png")
    to_pil_image(grid).save(grid_path, quality=95)
    print(f"[Eval] Full comparison grid → {grid_path}")

    # ─── Zoomed-in text crops ────────────────────────────────────────────
    # Crop a 128×128 region from the upper-center (where text usually is)
    crop_size = 128
    h, w = originals.shape[2], originals.shape[3]
    top = h // 6      # Upper portion where receipt headers typically are
    left = w // 4     # Centered
    bottom = top + crop_size
    right = left + crop_size

    orig_crops = originals[:, :, top:bottom, left:right]
    recon_crops = reconstructed[:, :, top:bottom, left:right]

    crop_comparison = torch.stack([orig_crops, recon_crops], dim=1)
    crop_comparison = crop_comparison.view(-1, *orig_crops.shape[1:])
    crop_grid = vutils.make_grid(crop_comparison, nrow=2, padding=2, pad_value=0.8)
    crop_path = os.path.join(OUTPUT_DIR, "eval_text_zoom_crops.png")
    to_pil_image(crop_grid).save(crop_path, quality=95)
    print(f"[Eval] Text zoom crops → {crop_path}")

    # ─── Individual image pairs ──────────────────────────────────────────
    pairs_dir = os.path.join(OUTPUT_DIR, "eval_pairs")
    os.makedirs(pairs_dir, exist_ok=True)

    for i in range(min(num_samples, 6)):
        orig_img = to_pil_image(originals[i])
        recon_img = to_pil_image(reconstructed[i])

        # Side by side
        combined = Image.new("RGB", (orig_img.width * 2 + 10, orig_img.height), (128, 128, 128))
        combined.paste(orig_img, (0, 0))
        combined.paste(recon_img, (orig_img.width + 10, 0))
        combined.save(os.path.join(pairs_dir, f"pair_{i:02d}.png"), quality=95)

    print(f"[Eval] Individual pairs → {pairs_dir}/")


def evaluate():
    """Run full evaluation pipeline."""
    print("=" * 70)
    print("  Document Autoencoder — Evaluation")
    print("=" * 70)
    print(f"  Device: {DEVICE}")
    print()

    # Load model
    model, checkpoint = load_best_model()

    # Get data
    _, val_loader = get_dataloaders()

    # Compute metrics
    ssim_arr, psnr_arr = evaluate_metrics(model, val_loader, DEVICE)

    # Print summary table
    print()
    print("=" * 50)
    print("  METRIC SUMMARY")
    print("=" * 50)
    print(f"  {'Metric':<20} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    print(f"  {'SSIM':<20} {ssim_arr.mean():>10.4f} {ssim_arr.std():>10.4f} {ssim_arr.min():>10.4f} {ssim_arr.max():>10.4f}")
    print(f"  {'PSNR (dB)':<20} {psnr_arr.mean():>10.2f} {psnr_arr.std():>10.2f} {psnr_arr.min():>10.2f} {psnr_arr.max():>10.2f}")
    print("=" * 50)
    print(f"  Total images evaluated: {len(ssim_arr)}")
    print(f"  Training epoch: {checkpoint['epoch']}")
    print()

    # Save comparison images
    save_comparison_results(model, val_loader, DEVICE)

    print()
    print("  ✓ Evaluation complete! Check outputs/ for visual results.")
    print()


if __name__ == "__main__":
    evaluate()
