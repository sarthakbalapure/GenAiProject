import os
import torch
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import sys

# Ensure project root is in path
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from autoencoder.model import DocumentVAE
from autoencoder.dataset import get_dataloaders
from autoencoder.config import CHECKPOINT_DIR, DEVICE, OUTPUT_DIR

def generate_samples(num_samples=3):
    print("Loading VAE model...")
    model = DocumentVAE().to(DEVICE)
    best_ckpt = os.path.join(CHECKPOINT_DIR, 'best_vae.pth')
    
    if not os.path.exists(best_ckpt):
        print(f"Checkpoint not found at {best_ckpt}. Please train the model first.")
        return

    # Load weights
    ckpt = torch.load(best_ckpt, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print("Model loaded.")

    print("Fetching a validation image...")
    _, val_loader = get_dataloaders()
    inputs, targets = next(iter(val_loader))
    
    # Take the first image from the batch
    single_img = inputs[0:1].to(DEVICE)

    samples = []
    print(f"Generating {num_samples} samples from the given image latent space...")
    with torch.no_grad():
        # Using forward with sample_mode=True to ensure pure latent space generation
        # without being forced to replicate exactly due to the skip connections.
        
        # Alternatively, we can use encode_vae if we want to manually reparameterize:
        mu, log_var, skips = model.encode_vae(single_img)
        
        # Sample multiple times from the latent distribution
        for i in range(num_samples):
            z = model.reparameterize(mu, log_var)
            
            # Note: since the VAE uses skip connections from the original image,
            # we need to zero them out, otherwise it will just identically reconstruct 
            # the original image instead of generating variations.
            zeroed_skips = tuple(torch.zeros_like(s) for s in skips)
            
            out = model.decode(z, zeroed_skips)
            samples.append(out.cpu())

    print("Saving visualization...")
    all_imgs = torch.cat([single_img.cpu()] + samples, dim=0)
    grid = vutils.make_grid(all_imgs, nrow=num_samples + 1, padding=4, pad_value=1.0)
    
    plt.figure(figsize=(5 * (num_samples + 1), 5))
    plt.imshow(grid.permute(1, 2, 0).numpy())
    
    titles = ["Original Input"] + [f"Generated Sample {i+1}" for i in range(num_samples)]
    
    plt.title(" | ".join(titles), fontsize=14)
    plt.axis("off")
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, 'VAEOutputs', 'generated_samples.png')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    print(f"Done! Saved grid to {output_path}")

if __name__ == "__main__":
    generate_samples(3)
