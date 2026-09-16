"""
Training loop for the VAE autoencoder.
"""

import os
import json
import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.utils as vutils
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm

from autoencoder.config import (
    DEVICE, CHECKPOINT_DIR, OUTPUT_DIR,
    LEARNING_RATE, WEIGHT_DECAY, VAE_EPOCHS, VAE_BETA,
    GRAD_CLIP_MAX_NORM, SCHEDULER_T0, SCHEDULER_T_MULT,
    EARLY_STOP_PATIENCE, SAVE_GRID_EVERY, LOG_INTERVAL,
)
from autoencoder.model import DocumentVAE, count_parameters
from autoencoder.losses import VAELoss
from autoencoder.dataset import get_dataloaders

# Make VAE outputs dir
VAE_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "VAEOutputs")
os.makedirs(VAE_OUTPUT_DIR, exist_ok=True)


def save_vae_samples(model, val_loader, epoch, device, num_samples=1):
    """
    Save grid with: Original | Sample 1 | Sample 2 | Sample 3
    """
    model.eval()
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # We just need one image to generate samples from
            single_input = inputs[0:1]
            single_target = targets[0:1]
            
            mu, logvar, skips = model.encode_vae(single_input)
            
            # ZERO OUT SKIPS: We must do this so the decoder is forced to look at the 
            # latent space (z) instead of just copying the input image!
            empty_skips = tuple(torch.zeros_like(s) for s in skips)
            
            # We can use a normal temperature now since the skips aren't overpowering it
            temperature = 1.0
            
            sample_1 = model.decode(model.reparameterize(mu, logvar, temperature), empty_skips)
            sample_2 = model.decode(model.reparameterize(mu, logvar, temperature), empty_skips)
            sample_3 = model.decode(model.reparameterize(mu, logvar, temperature), empty_skips)
            
            comparison = torch.cat([single_target, sample_1, sample_2, sample_3], dim=0)
            
            grid = vutils.make_grid(comparison, nrow=4, padding=4, pad_value=0.5)
            grid_path = os.path.join(VAE_OUTPUT_DIR, f"vae_samples_epoch_{epoch:03d}.png")
            to_pil_image(grid).save(grid_path)
            print(f"  [Grid] Saved VAE samples grid -> {grid_path}")
            break

def save_latent_traversal(model, val_loader, epoch, device, steps=6):
    """
    Interpolate between two documents in latent space to show smooth transitions.
    (Research paper level traversal plot).
    """
    model.eval()
    with torch.no_grad():
        # Get two batches to ensure we have different images
        it = iter(val_loader)
        inputs1, _ = next(it)
        try:
            inputs2, _ = next(it)
        except StopIteration:
            inputs2 = inputs1[::-1] # Fallback if dataset is too small
            
        img1 = inputs1[0:1].to(device)
        img2 = inputs2[0:1].to(device)
        
        mu1, logvar1, skips1 = model.encode_vae(img1)
        mu2, logvar2, _ = model.encode_vae(img2)
        
        # ZERO OUT SKIPS: Force it to interpolate in pure latent space
        empty_skips = tuple(torch.zeros_like(s) for s in skips1)
        
        traversal_imgs = []
        for i in range(steps):
            alpha = i / (steps - 1)
            # Linear interpolation in latent space
            z_interp = (1.0 - alpha) * mu1 + alpha * mu2
            img_interp = model.decode(z_interp, empty_skips)
            traversal_imgs.append(img_interp)
            
        comparison = torch.cat(traversal_imgs, dim=0)
        grid = vutils.make_grid(comparison, nrow=steps, padding=4, pad_value=0.5)
        grid_path = os.path.join(VAE_OUTPUT_DIR, f"latent_traversal_epoch_{epoch:03d}.png")
        to_pil_image(grid).save(grid_path)
        print(f"  [Grid] Saved Latent Traversal -> {grid_path}")

def train_one_epoch(model, train_loader, loss_fn, optimizer, device, epoch):
    model.train()
    total_losses = {"total_vae": 0, "kl": 0, "total": 0, "ssim_l1": 0}
    num_batches = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch:3d} [Train]", leave=False)
    for inputs, targets in pbar:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        out, mu, logvar = model(inputs)
        losses = loss_fn(out, targets, mu, logvar)

        losses["total_vae"].backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_MAX_NORM)
        optimizer.step()

        total_losses["total_vae"] += losses["total_vae"].item()
        total_losses["kl"] += losses["kl"]
        total_losses["total"] += losses["total"].item()
        total_losses["ssim_l1"] += losses["ssim_l1"]
        num_batches += 1

        pbar.set_postfix({"loss": f"{losses['total_vae'].item():.4f}", "kl": f"{losses['kl']:.4f}"})

    return {k: v / num_batches for k, v in total_losses.items()}

@torch.no_grad()
def validate(model, val_loader, loss_fn, device):
    model.eval()
    total_losses = {"total_vae": 0, "kl": 0, "ssim_raw": 0, "total": 0}
    num_batches = 0

    for inputs, targets in val_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        out, mu, logvar = model(inputs)

        losses = loss_fn(out, targets, mu, logvar)

        total_losses["total_vae"] += losses["total_vae"].item()
        total_losses["kl"] += losses["kl"]
        total_losses["total"] += losses["total"].item()
        total_losses["ssim_raw"] += losses["ssim_raw"]
        num_batches += 1

    return {k: v / num_batches for k, v in total_losses.items()}

def train():
    print("=" * 70)
    print("  Document VAE Training")
    print("=" * 70)
    
    train_loader, val_loader = get_dataloaders()
    model = DocumentVAE().to(DEVICE)
    loss_fn = VAELoss(beta=VAE_BETA).to(DEVICE)
    
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=VAE_EPOCHS)
    
    best_ssim = 0.0
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_ssim": [], "kl_loss": []}

    for epoch in range(1, VAE_EPOCHS + 1):
        t0 = time.time()
        
        # KL Annealing: Linearly increase beta from 0 to VAE_BETA over 15 epochs
        anneal_epochs = 15.0
        current_beta = VAE_BETA * min(1.0, (epoch - 1) / anneal_epochs)
        loss_fn.beta = current_beta

        train_losses = train_one_epoch(model, train_loader, loss_fn, optimizer, DEVICE, epoch)
        scheduler.step()

        val_losses = validate(model, val_loader, loss_fn, DEVICE)
        val_ssim = val_losses["ssim_raw"]
        
        elapsed = time.time() - t0
        
        history["train_loss"].append(train_losses["total_vae"])
        history["val_loss"].append(val_losses["total_vae"])
        history["val_ssim"].append(val_ssim)
        history["kl_loss"].append(val_losses["kl"])

        print(
            f"  Epoch {epoch:3d}/{VAE_EPOCHS} | "
            f"Train Loss: {train_losses['total_vae']:.4f} (KL: {train_losses['kl']:.4f}) | "
            f"Val Loss: {val_losses['total_vae']:.4f} (KL: {val_losses['kl']:.4f}) | "
            f"Val SSIM: {val_ssim:.4f} | "
            f"{elapsed:.1f}s"
        )

        with open(os.path.join(OUTPUT_DIR, "vae_training_status.json"), "w") as f:
            json.dump({
                "epoch": epoch,
                "total_epochs": VAE_EPOCHS,
                "train_loss": train_losses['total_vae'],
                "train_kl": train_losses['kl'],
                "val_loss": val_losses['total_vae'],
                "val_kl": val_losses['kl'],
                "val_ssim": val_ssim,
                "elapsed": elapsed,
                "history": history
            }, f)

        if epoch % SAVE_GRID_EVERY == 0 or epoch == 1:
            save_vae_samples(model, val_loader, epoch, DEVICE)
            save_latent_traversal(model, val_loader, epoch, DEVICE)

        if val_ssim > best_ssim:
            best_ssim = val_ssim
            patience_counter = 0
            ckpt_path = os.path.join(CHECKPOINT_DIR, "best_vae.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "best_ssim": best_ssim,
                "history": history
            }, ckpt_path)
            print(f"  * New best SSIM: {best_ssim:.4f} - checkpoint saved!")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"Early stopping after {EARLY_STOP_PATIENCE} epochs.")
                break
                
        # Always save the latest checkpoint for resuming
        latest_ckpt_path = os.path.join(CHECKPOINT_DIR, "latest_vae.pth")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_ssim": best_ssim,
            "history": history
        }, latest_ckpt_path)
                
if __name__ == "__main__":
    train()
