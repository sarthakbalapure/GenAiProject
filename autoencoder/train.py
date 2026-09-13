"""
Training loop for the document autoencoder.

Features:
  - AdamW optimizer with CosineAnnealingWarmRestarts scheduler
  - Gradient clipping for stability
  - Early stopping on validation SSIM
  - Reconstruction comparison grids saved periodically
  - Best checkpoint saved by val SSIM
"""

import os
import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torchvision.utils as vutils
from tqdm import tqdm

from autoencoder.config import (
    DEVICE, CHECKPOINT_DIR, OUTPUT_DIR,
    LEARNING_RATE, WEIGHT_DECAY, EPOCHS,
    GRAD_CLIP_MAX_NORM, SCHEDULER_T0, SCHEDULER_T_MULT,
    EARLY_STOP_PATIENCE, SAVE_GRID_EVERY, LOG_INTERVAL,
)
from autoencoder.model import DocumentAutoencoder, count_parameters
from autoencoder.losses import CompositeLoss
from autoencoder.dataset import get_dataloaders


def save_comparison_grid(model, val_loader, epoch, device, num_samples=6):
    """
    Save a side-by-side comparison grid: original | reconstructed.
    Saved to outputs/ directory for visual inspection.
    """
    model.eval()
    originals = []
    reconstructed = []

    with torch.no_grad():
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

    # Interleave: original, reconstructed, original, reconstructed, ...
    comparison = torch.stack([originals, reconstructed], dim=1)
    comparison = comparison.view(-1, *originals.shape[1:])

    grid = vutils.make_grid(comparison, nrow=2, padding=4, pad_value=0.5)
    grid_path = os.path.join(OUTPUT_DIR, f"reconstruction_epoch_{epoch:03d}.png")

    # Convert to PIL and save
    from torchvision.transforms.functional import to_pil_image
    to_pil_image(grid).save(grid_path)
    print(f"  [Grid] Saved comparison grid → {grid_path}")


def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute Peak Signal-to-Noise Ratio (in dB)."""
    mse = torch.mean((pred - target) ** 2).item()
    if mse < 1e-10:
        return 100.0
    return 10 * torch.log10(torch.tensor(1.0 / mse)).item()


def train_one_epoch(model, train_loader, loss_fn, optimizer, device, epoch):
    """Train for one epoch, return average losses."""
    model.train()
    total_losses = {"total": 0, "ssim_l1": 0, "perceptual": 0, "edge": 0}
    num_batches = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch:3d} [Train]", leave=False)
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        losses = loss_fn(outputs, targets)

        losses["total"].backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_MAX_NORM)
        optimizer.step()

        # Accumulate losses
        total_losses["total"] += losses["total"].item()
        total_losses["ssim_l1"] += losses["ssim_l1"]
        total_losses["perceptual"] += losses["perceptual"]
        total_losses["edge"] += losses["edge"]
        num_batches += 1

        pbar.set_postfix({
            "loss": f"{losses['total'].item():.4f}",
            "ssim": f"{losses['ssim_raw']:.4f}",
        })

    # Average over batches
    return {k: v / num_batches for k, v in total_losses.items()}


@torch.no_grad()
def validate(model, val_loader, loss_fn, device):
    """Validate on the val set, return average losses + SSIM + PSNR."""
    model.eval()
    total_losses = {"total": 0, "ssim_l1": 0, "perceptual": 0, "edge": 0, "ssim_raw": 0}
    total_psnr = 0
    num_batches = 0

    for inputs, targets in val_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs)

        losses = loss_fn(outputs, targets)

        total_losses["total"] += losses["total"].item()
        total_losses["ssim_l1"] += losses["ssim_l1"]
        total_losses["perceptual"] += losses["perceptual"]
        total_losses["edge"] += losses["edge"]
        total_losses["ssim_raw"] += losses["ssim_raw"]
        total_psnr += compute_psnr(outputs, targets)
        num_batches += 1

    avg = {k: v / num_batches for k, v in total_losses.items()}
    avg["psnr"] = total_psnr / num_batches
    return avg


def train():
    """Main training function."""
    print("=" * 70)
    print("  Document Autoencoder — Sharp Reconstruction Training")
    print("=" * 70)
    print(f"  Device:        {DEVICE}")
    print(f"  Image size:    512×512")
    print(f"  Epochs:        {EPOCHS}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Batch size:    see config")
    print(f"  Loss:          SSIM+L1 + VGG Perceptual + Sobel Edge")
    print("=" * 70)

    # ─── Data ────────────────────────────────────────────────────────────
    train_loader, val_loader = get_dataloaders()

    # ─── Model ───────────────────────────────────────────────────────────
    model = DocumentAutoencoder().to(DEVICE)
    print(f"\n  Model parameters: {count_parameters(model):,}")

    # ─── Loss ────────────────────────────────────────────────────────────
    loss_fn = CompositeLoss().to(DEVICE)

    # ─── Optimizer & Scheduler ───────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=SCHEDULER_T0, T_mult=SCHEDULER_T_MULT)

    # ─── Training State ──────────────────────────────────────────────────
    best_ssim = 0.0
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_ssim": [], "val_psnr": []}

    print(f"\n  Starting training...\n")

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        # Train
        train_losses = train_one_epoch(model, train_loader, loss_fn, optimizer, DEVICE, epoch)
        scheduler.step()

        # Validate
        val_losses = validate(model, val_loader, loss_fn, DEVICE)
        val_ssim = val_losses["ssim_raw"]
        val_psnr = val_losses["psnr"]

        elapsed = time.time() - t0

        # Log
        history["train_loss"].append(train_losses["total"])
        history["val_loss"].append(val_losses["total"])
        history["val_ssim"].append(val_ssim)
        history["val_psnr"].append(val_psnr)

        print(
            f"  Epoch {epoch:3d}/{EPOCHS} │ "
            f"Train Loss: {train_losses['total']:.4f} │ "
            f"Val Loss: {val_losses['total']:.4f} │ "
            f"Val SSIM: {val_ssim:.4f} │ "
            f"Val PSNR: {val_psnr:.2f} dB │ "
            f"LR: {optimizer.param_groups[0]['lr']:.6f} │ "
            f"{elapsed:.1f}s"
        )
        print(
            f"           │ "
            f"SSIM+L1: {val_losses['ssim_l1']:.4f} │ "
            f"Perceptual: {val_losses['perceptual']:.4f} │ "
            f"Edge: {val_losses['edge']:.4f}"
        )

        # Save comparison grid
        if epoch % SAVE_GRID_EVERY == 0 or epoch == 1:
            save_comparison_grid(model, val_loader, epoch, DEVICE)

        # Check for improvement
        if val_ssim > best_ssim:
            best_ssim = val_ssim
            patience_counter = 0
            ckpt_path = os.path.join(CHECKPOINT_DIR, "best_autoencoder.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_ssim": best_ssim,
                "val_psnr": val_psnr,
                "history": history,
            }, ckpt_path)
            print(f"  ★ New best SSIM: {best_ssim:.4f} — checkpoint saved!")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"\n  ⚠ Early stopping after {EARLY_STOP_PATIENCE} epochs without improvement.")
                break

        print()

    # Save final checkpoint
    final_path = os.path.join(CHECKPOINT_DIR, "final_autoencoder.pth")
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_ssim": best_ssim,
        "history": history,
    }, final_path)

    print("=" * 70)
    print(f"  Training complete!")
    print(f"  Best Val SSIM: {best_ssim:.4f}")
    print(f"  Checkpoints:   {CHECKPOINT_DIR}")
    print(f"  Output grids:  {OUTPUT_DIR}")
    print("=" * 70)

    return model, history


if __name__ == "__main__":
    train()
