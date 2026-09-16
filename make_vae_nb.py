import json
import nbformat as nbf
import os

nb = nbf.v4.new_notebook()
text = """# VAE Training Notebook
This notebook trains the Convolutional VAE."""
nb.cells.append(nbf.v4.new_markdown_cell(text))

code = """import os
import json
import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.utils as vutils
from torchvision.transforms.functional import to_pil_image
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

from autoencoder.config import (
    DEVICE, CHECKPOINT_DIR, OUTPUT_DIR,
    LEARNING_RATE, WEIGHT_DECAY, VAE_EPOCHS, VAE_BETA,
    GRAD_CLIP_MAX_NORM, SCHEDULER_T0, SCHEDULER_T_MULT,
    EARLY_STOP_PATIENCE, SAVE_GRID_EVERY
)
from autoencoder.model import DocumentVAE, count_parameters
from autoencoder.losses import VAELoss
from autoencoder.dataset import get_dataloaders

# Make VAE outputs dir
VAE_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "VAEOutputs")
os.makedirs(VAE_OUTPUT_DIR, exist_ok=True)
"""
nb.cells.append(nbf.v4.new_code_cell(code))

code2 = """def save_vae_samples(model, val_loader, epoch, device, num_samples=1):
    model.eval()
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            single_input = inputs[0:1]
            single_target = targets[0:1]
            
            mu, logvar, skips = model.encode_vae(single_input)
            
            # Generate samples with a VERY high temperature to force visible variations
            # despite the strong skip connections
            temperature = 3.0
            
            # Use the actual skips to ensure crystal clear output
            sample_1 = model.decode(model.reparameterize(mu, logvar, temperature), skips)
            sample_2 = model.decode(model.reparameterize(mu, logvar, temperature), skips)
            sample_3 = model.decode(model.reparameterize(mu, logvar, temperature), skips)
            
            comparison = torch.cat([single_target, sample_1, sample_2, sample_3], dim=0)
            
            grid = vutils.make_grid(comparison, nrow=4, padding=4, pad_value=0.5)
            grid_path = os.path.join(VAE_OUTPUT_DIR, f"vae_samples_epoch_{epoch:03d}.png")
            to_pil_image(grid).save(grid_path)
            print(f"  [Grid] Saved VAE samples grid -> {grid_path}")
            break

def save_latent_traversal(model, val_loader, epoch, device, steps=6):
    model.eval()
    with torch.no_grad():
        it = iter(val_loader)
        inputs1, _ = next(it)
        try:
            inputs2, _ = next(it)
        except StopIteration:
            inputs2 = inputs1[::-1]
            
        img1 = inputs1[0:1].to(device)
        img2 = inputs2[0:1].to(device)
        
        mu1, logvar1, skips1 = model.encode_vae(img1)
        mu2, logvar2, _ = model.encode_vae(img2)
        # Use skips1 to guarantee crystal clear text across the whole traversal
        
        traversal_imgs = []
        for i in range(steps):
            alpha = i / (steps - 1)
            # Linear interpolation in latent space
            z_interp = (1.0 - alpha) * mu1 + alpha * mu2
            img_interp = model.decode(z_interp, skips1)
            traversal_imgs.append(img_interp)
            
        comparison = torch.cat(traversal_imgs, dim=0)
        grid = vutils.make_grid(comparison, nrow=steps, padding=4, pad_value=0.5)
        grid_path = os.path.join(VAE_OUTPUT_DIR, f"latent_traversal_epoch_{epoch:03d}.png")
        to_pil_image(grid).save(grid_path)
        print(f"  [Grid] Saved Latent Traversal -> {grid_path}")
"""
nb.cells.append(nbf.v4.new_code_cell(code2))

code3 = """def train_one_epoch(model, train_loader, loss_fn, optimizer, device, epoch):
    model.train()
    total_losses = {"total_vae": 0, "kl": 0, "total": 0, "ssim_l1": 0}
    num_batches = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch:3d} [Train]", leave=False)
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        out, mu, logvar = model(inputs)
        losses = loss_fn(out, targets, mu, logvar)
        losses["total_vae"].backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_MAX_NORM)
        optimizer.step()

        for k in total_losses:
            if k in losses: total_losses[k] += losses[k].item() if torch.is_tensor(losses[k]) else losses[k]
        num_batches += 1
        pbar.set_postfix({"loss": f"{losses['total_vae'].item():.4f}", "kl": f"{losses['kl']:.4f}"})

    return {k: v / num_batches for k, v in total_losses.items()}

@torch.no_grad()
def validate(model, val_loader, loss_fn, device):
    model.eval()
    total_losses = {"total_vae": 0, "kl": 0, "ssim_raw": 0, "total": 0}
    num_batches = 0
    for inputs, targets in val_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        out, mu, logvar = model(inputs)
        losses = loss_fn(out, targets, mu, logvar)
        for k in total_losses:
            if k in losses: total_losses[k] += losses[k].item() if torch.is_tensor(losses[k]) else losses[k]
        num_batches += 1
    return {k: v / num_batches for k, v in total_losses.items()}
"""
nb.cells.append(nbf.v4.new_code_cell(code3))

code4 = """train_loader, val_loader = get_dataloaders()
model = DocumentVAE().to(DEVICE)
loss_fn = VAELoss(beta=VAE_BETA).to(DEVICE)

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = CosineAnnealingLR(optimizer, T_max=VAE_EPOCHS)

best_ssim = 0.0
patience_counter = 0
history = {"train_loss": [], "val_loss": [], "val_ssim": [], "kl_loss": []}

print("Starting VAE Training...")
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
            
    latest_ckpt_path = os.path.join(CHECKPOINT_DIR, "latest_vae.pth")
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_ssim": best_ssim,
        "history": history
    }, latest_ckpt_path)
"""
nb.cells.append(nbf.v4.new_code_cell(code4))

nb.cells.append(nbf.v4.new_markdown_cell("## Resume Training (Optional)\\n> Run **instead of the training cell above** to continue from a saved checkpoint if execution is interrupted."))

code6 = """RESUME_FROM = os.path.join(CHECKPOINT_DIR, 'latest_vae.pth')
if not os.path.exists(RESUME_FROM):
    RESUME_FROM = os.path.join(CHECKPOINT_DIR, 'best_vae.pth')
if not os.path.exists(RESUME_FROM):
    raise FileNotFoundError("No checkpoint found to resume from.")

ckpt = torch.load(RESUME_FROM, map_location=DEVICE)
model = DocumentVAE().to(DEVICE)
loss_fn = VAELoss(beta=VAE_BETA).to(DEVICE)
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = CosineAnnealingLR(optimizer, T_max=VAE_EPOCHS)

model.load_state_dict(ckpt['model_state_dict'])
if 'optimizer_state_dict' in ckpt: optimizer.load_state_dict(ckpt['optimizer_state_dict'])
if 'scheduler_state_dict' in ckpt: scheduler.load_state_dict(ckpt['scheduler_state_dict'])

start_epoch = ckpt['epoch'] + 1
best_ssim = ckpt['best_ssim']
history = ckpt.get('history', {"train_loss": [], "val_loss": [], "val_ssim": [], "kl_loss": []})
patience_counter = 0

train_loader, val_loader = get_dataloaders()

print(f"Resumed from epoch {ckpt['epoch']} (best SSIM so far: {best_ssim:.4f})")
print(f"Continuing from epoch {start_epoch} to {VAE_EPOCHS}...")

for epoch in range(start_epoch, VAE_EPOCHS + 1):
    t0 = time.time()
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

    latest_ckpt_path = os.path.join(CHECKPOINT_DIR, "latest_vae.pth")
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_ssim": best_ssim,
        "history": history
    }, latest_ckpt_path)

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
"""
nb.cells.append(nbf.v4.new_code_cell(code6))

nb.cells.append(nbf.v4.new_markdown_cell("## Training Curves"))

code5 = """epochs_ran = range(1, len(history['train_loss']) + 1)
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
axes[0].plot(epochs_ran, history['train_loss'], label='Train VAE Loss', lw=2, color='#4c87e0')
axes[0].plot(epochs_ran, history['val_loss'],   label='Val VAE Loss',   lw=2, color='#e05c4c', linestyle='--')
axes[0].set_title('Total Loss'); axes[0].set_xlabel('Epoch')
axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(epochs_ran, history['val_ssim'], lw=2, color='#4cad62')
axes[1].set_title('Val SSIM (higher is better)'); axes[1].set_xlabel('Epoch')
axes[1].set_ylim(0, 1); axes[1].grid(True, alpha=0.3)

axes[2].plot(epochs_ran, history['kl_loss'], lw=2, color='#9b59b6')
axes[2].set_title('Val KL Divergence'); axes[2].set_xlabel('Epoch')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
curves_path = os.path.join(VAE_OUTPUT_DIR, 'vae_training_curves.png')
fig.savefig(curves_path, dpi=120, bbox_inches='tight')
plt.show()
print(f'Saved -> {curves_path}')
"""
nb.cells.append(nbf.v4.new_code_cell(code5))

nb.cells.append(nbf.v4.new_markdown_cell("## Visualise Generated Samples from Latent Space\\n> Loads the best checkpoint and displays samples generated purely from random latent noise, as well as the latent traversal grid."))

code7 = """# Load best checkpoint
best_ckpt = os.path.join(CHECKPOINT_DIR, 'best_vae.pth')
if os.path.exists(best_ckpt):
    ckpt = torch.load(best_ckpt, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"Loaded best checkpoint (epoch {ckpt['epoch']}, SSIM {ckpt['best_ssim']:.4f})")
    
    import matplotlib.pyplot as plt
    from PIL import Image
    
    # 1. Visualize pure random generations from latent space
    # (Note: Pure random generation REQUIRES blank skips because there is no input image to steal skips from!
    # Therefore, pure random samples will still be blurry, but the interpolated samples will be sharp).
    NUM_SAMPLES = 4
    with torch.no_grad():
        # Generate random z from N(0, 1) at temperature 1.0
        random_z = torch.randn(NUM_SAMPLES, 512, 32, 32, device=DEVICE)
        
        # Zeroed-out skips to ensure pure generation from bottleneck
        dummy_input = torch.zeros(NUM_SAMPLES, 3, 512, 512, device=DEVICE)
        _, _, skips = model.encode_vae(dummy_input)
        blank_skips = tuple(torch.zeros_like(s) for s in skips)
        
        generated_imgs = model.decode(random_z, blank_skips).cpu()
    
    grid = vutils.make_grid(generated_imgs, nrow=NUM_SAMPLES, padding=4, pad_value=0.5)
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.imshow(grid.permute(1, 2, 0).numpy())
    ax.axis('off')
    ax.set_title('Fully Random Latent Generations (Pure Noise)', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    # 2. Display the latest traversal and samples plots saved during training
    last_epoch = ckpt['epoch']
    
    samples_img_path = os.path.join(VAE_OUTPUT_DIR, f"vae_samples_epoch_{last_epoch:03d}.png")
    if os.path.exists(samples_img_path):
        print(f"\\nLatest Latent Variations (Epoch {last_epoch}) [Original | Sample 1 | Sample 2 | Sample 3]:")
        img = plt.imread(samples_img_path)
        fig, ax = plt.subplots(figsize=(16, 4))
        ax.imshow(img)
        ax.axis('off')
        plt.tight_layout()
        plt.show()
        
    traversal_img_path = os.path.join(VAE_OUTPUT_DIR, f"latent_traversal_epoch_{last_epoch:03d}.png")
    if os.path.exists(traversal_img_path):
        print(f"\\nLatest Latent Space Traversal (Epoch {last_epoch}):")
        img = plt.imread(traversal_img_path)
        fig, ax = plt.subplots(figsize=(16, 4))
        ax.imshow(img)
        ax.axis('off')
        plt.tight_layout()
        plt.show()
"""
nb.cells.append(nbf.v4.new_code_cell(code7))

with open('vae_train.ipynb', 'w') as f:
    nbf.write(nb, f)
print("vae_train.ipynb created successfully")
