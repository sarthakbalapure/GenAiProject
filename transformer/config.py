"""
Centralized configuration for the LayoutLM-based Document Transformer.
All hyperparameters and paths in one place for easy tuning.
"""

import os
import torch

# PyTorch workaround for cuBLAS execution failures on Windows/RTX
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# ─── Paths ───────────────────────────────────────────────────────────────────
PROJECT_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR      = os.path.join(PROJECT_ROOT, "ICDAR-2019-SROIE", "data")
IMG_DIR       = os.path.join(DATA_DIR, "img")
BOX_DIR       = os.path.join(DATA_DIR, "box")
KEY_DIR       = os.path.join(DATA_DIR, "key")

CHECKPOINT_DIR     = os.path.join(PROJECT_ROOT, "checkpoints")
OUTPUT_DIR         = os.path.join(PROJECT_ROOT, "outputs")
TRANSFORMER_CKPT   = os.path.join(CHECKPOINT_DIR, "transformer_model")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TRANSFORMER_CKPT, exist_ok=True)

# ─── Device ──────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Pretrained model ────────────────────────────────────────────────────────
# LayoutLM-base is lightweight and fine-tunes well on small datasets (~600 images).
# Switch to "microsoft/layoutlmv2-base-uncased" for image-feature-aware attention.
MODEL_NAME = "microsoft/layoutlm-base-uncased"

# ─── Label schema (BIO tagging for SROIE key fields) ─────────────────────────
LABELS = [
    "O",
    "B-COMPANY",  "I-COMPANY",
    "B-DATE",     "I-DATE",
    "B-ADDRESS",  "I-ADDRESS",
    "B-TOTAL",    "I-TOTAL",
]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for i, label in enumerate(LABELS)}
NUM_LABELS = len(LABELS)

# ─── Training Hyperparameters ────────────────────────────────────────────────
EPOCHS               = 30
BATCH_SIZE           = 2
LEARNING_RATE        = 2e-5         # Standard AdamW LR for fine-tuning transformers
WEIGHT_DECAY         = 0.01
WARMUP_RATIO         = 0.1          # Warmup for 10% of total training steps
MAX_SEQ_LENGTH       = 512
VAL_RATIO            = 0.2          # 80/20 train/val split
RANDOM_SEED          = 42

# ─── Early Stopping ─────────────────────────────────────────────────────────
EARLY_STOP_PATIENCE  = 3            # Stop if val F1 doesn't improve for 3 epochs

# ─── Logging ─────────────────────────────────────────────────────────────────
LOGGING_STEPS        = 50           # Log every N optimizer steps
EVAL_STRATEGY        = "epoch"      # Evaluate and checkpoint every epoch
SAVE_TOTAL_LIMIT     = 3            # Keep at most N checkpoints
METRIC_FOR_BEST      = "f1"
