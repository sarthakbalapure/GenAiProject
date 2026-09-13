"""
SROIE Dataset loader for autoencoder training.
Loads receipt images, resizes with aspect-ratio-preserving padding,
and splits into train/val sets.
"""

import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from autoencoder.config import (
    DATA_DIR, IMG_SIZE, VAL_RATIO, RANDOM_SEED,
    BATCH_SIZE, NUM_WORKERS
)


class ResizeWithPad:
    """
    Resize an image to (target_size × target_size) while preserving aspect ratio.
    Pads with white (255) to match receipt background.
    """
    def __init__(self, target_size: int):
        self.target_size = target_size

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        scale = self.target_size / max(w, h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        # Resize with high-quality Lanczos resampling
        img = img.resize((new_w, new_h), Image.LANCZOS)

        # Create white canvas and paste resized image centered
        canvas = Image.new("RGB", (self.target_size, self.target_size), (255, 255, 255))
        paste_x = (self.target_size - new_w) // 2
        paste_y = (self.target_size - new_h) // 2
        canvas.paste(img, (paste_x, paste_y))

        return canvas


class SROIEDataset(Dataset):
    """
    Dataset for SROIE receipt images.
    Returns normalized [0, 1] tensors of shape (3, IMG_SIZE, IMG_SIZE).
    For autoencoder: input = target (self-supervised reconstruction).
    """
    def __init__(self, image_paths: list, is_train: bool = True):
        self.image_paths = image_paths
        self.is_train = is_train
        self.resize_pad = ResizeWithPad(IMG_SIZE)

        # Training augmentation: very subtle brightness/contrast jitter
        # (no geometric transforms — we don't want to distort text)
        self.color_jitter = T.ColorJitter(
            brightness=0.05,
            contrast=0.05,
            saturation=0.02,
            hue=0.01,
        )

        self.to_tensor = T.ToTensor()  # Converts [0,255] PIL → [0,1] tensor

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")

        # Resize with padding
        img = self.resize_pad(img)

        if self.is_train:
            # Apply subtle color jitter for augmentation
            img_augmented = self.color_jitter(img)
            # Input is augmented, target is clean — teaches denoising too
            input_tensor = self.to_tensor(img_augmented)
            target_tensor = self.to_tensor(img)
        else:
            # Validation: input == target (pure reconstruction)
            input_tensor = self.to_tensor(img)
            target_tensor = self.to_tensor(img)

        return input_tensor, target_tensor


def get_image_paths() -> list:
    """Get all image file paths from the SROIE data directory, sorted."""
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    paths = []
    for fname in sorted(os.listdir(DATA_DIR)):
        ext = os.path.splitext(fname)[1].lower()
        if ext in valid_extensions:
            paths.append(os.path.join(DATA_DIR, fname))
    return paths


def get_dataloaders() -> tuple:
    """
    Create train and validation DataLoaders with deterministic split.
    Returns: (train_loader, val_loader)
    """
    all_paths = get_image_paths()
    print(f"[Dataset] Found {len(all_paths)} images in {DATA_DIR}")

    # Deterministic shuffle and split
    random.seed(RANDOM_SEED)
    shuffled = all_paths.copy()
    random.shuffle(shuffled)

    split_idx = int(len(shuffled) * (1 - VAL_RATIO))
    train_paths = shuffled[:split_idx]
    val_paths = shuffled[split_idx:]

    print(f"[Dataset] Train: {len(train_paths)} images, Val: {len(val_paths)} images")

    train_dataset = SROIEDataset(train_paths, is_train=True)
    val_dataset = SROIEDataset(val_paths, is_train=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader


if __name__ == "__main__":
    # Quick test: load one batch and print shapes
    train_loader, val_loader = get_dataloaders()
    for inputs, targets in train_loader:
        print(f"Input shape:  {inputs.shape}")   # (B, 3, 512, 512)
        print(f"Target shape: {targets.shape}")   # (B, 3, 512, 512)
        print(f"Value range:  [{inputs.min():.3f}, {inputs.max():.3f}]")
        break
