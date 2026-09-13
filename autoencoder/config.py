"""
Centralized configuration for the autoencoder training pipeline.
All hyperparameters in one place for easy tuning.
"""

import torch
import os

# ─── Paths ───────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "ICDAR-2019-SROIE", "data", "img")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

# Create directories if they don't exist
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Device ──────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Image Settings ──────────────────────────────────────────────────────────
IMG_SIZE = 512           # Resize to 512×512 for clearer outputs
IMG_CHANNELS = 3         # RGB

# ─── Data Split ──────────────────────────────────────────────────────────────
VAL_RATIO = 0.2          # 80% train, 20% val (~500 train, ~126 val)
RANDOM_SEED = 42         # Deterministic split

# ─── Training Hyperparameters ────────────────────────────────────────────────
BATCH_SIZE = 4           # Reduced to 4 to prevent CUDA OOM on 4GB VRAM
NUM_WORKERS = 0          # 0 avoids Windows multiprocessing IPC overhead
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
EPOCHS = 30
GRAD_CLIP_MAX_NORM = 1.0

# ─── Scheduler ───────────────────────────────────────────────────────────────
SCHEDULER_T0 = 10        # CosineAnnealingWarmRestarts period
SCHEDULER_T_MULT = 2     # Multiply period after each restart

# ─── Early Stopping ─────────────────────────────────────────────────────────
EARLY_STOP_PATIENCE = 15 # Stop if val SSIM doesn't improve for 15 epochs

# ─── Loss Weights ────────────────────────────────────────────────────────────
LAMBDA_SSIM_L1 = 1.0     # Weight for SSIM + L1 combined loss
LAMBDA_PERCEPTUAL = 0.5  # Weight for VGG perceptual loss
LAMBDA_EDGE = 0.5        # Weight for Sobel edge loss

# Within the SSIM+L1 term:
SSIM_WEIGHT = 0.84       # Relative weight of SSIM vs L1
L1_WEIGHT = 0.16         # Relative weight of L1 vs SSIM

# ─── Model Architecture ─────────────────────────────────────────────────────
ENCODER_CHANNELS = [3, 64, 128, 256, 512]   # Progressive channel widths
DECODER_CHANNELS = [512, 256, 128, 64, 32]  # Mirror + extra refinement block
LATENT_SPATIAL = IMG_SIZE // (2 ** 4)        # 512/16 = 32 (spatial size at bottleneck)

# ─── Logging ─────────────────────────────────────────────────────────────────
SAVE_GRID_EVERY = 1      # Save reconstruction comparison grids every N epochs
LOG_INTERVAL = 10        # Print batch-level loss every N batches
