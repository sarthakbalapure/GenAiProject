"""
U-Net style Convolutional Autoencoder for sharp document reconstruction.

Key anti-blur design choices:
  1. Skip connections — preserve high-frequency detail at every scale
  2. Upsample + Conv decoder — no checkerboard artifacts from ConvTranspose2d
  3. Conservative downsampling — 4 blocks only (512→32 spatial)
  4. Extra conv per block — deeper feature extraction without more downsampling
"""

import torch
import torch.nn as nn


class EncoderBlock(nn.Module):
    """
    Single encoder block: Conv(stride=2) → BN → LeakyReLU → Conv(stride=1) → BN → LeakyReLU
    The second conv adds depth without more spatial reduction.
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            # Downsample conv
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            # Refinement conv (same spatial size)
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DecoderBlock(nn.Module):
    """
    Single decoder block with skip connection input.
    Upsample(×2) → Concat(skip) → Conv → BN → LeakyReLU → Conv → BN → LeakyReLU

    Using nearest-neighbor upsample + conv instead of ConvTranspose2d
    to avoid checkerboard artifacts.
    """
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        # After concat: in_ch + skip_ch channels
        self.block = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            # Refinement conv
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        # Concatenate skip connection along channel dim
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


class DocumentAutoencoder(nn.Module):
    """
    U-Net style autoencoder for sharp document image reconstruction.

    Architecture:
        Encoder: 3→64→128→256→512 (4 downsampling blocks)
        Bottleneck: 512×32×32 (for 512×512 input)
        Decoder: 512→256→128→64→32 (4 upsampling blocks with skip connections)
        Output: Conv 32→3 + Sigmoid

    Skip connections from encoder to decoder at each resolution level
    are the critical element for preserving text sharpness.
    """
    def __init__(self):
        super().__init__()

        # ─── Encoder ─────────────────────────────────────────────────────
        # Input: (B, 3, 512, 512)
        self.enc1 = EncoderBlock(3, 64)      # → (B, 64, 256, 256)
        self.enc2 = EncoderBlock(64, 128)    # → (B, 128, 128, 128)
        self.enc3 = EncoderBlock(128, 256)   # → (B, 256, 64, 64)
        self.enc4 = EncoderBlock(256, 512)   # → (B, 512, 32, 32)

        # ─── Bottleneck ──────────────────────────────────────────────────
        # Extra processing at the bottleneck (no spatial change)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # ─── Decoder ─────────────────────────────────────────────────────
        # Each decoder block receives: upsampled features + skip connection
        self.dec4 = DecoderBlock(512, 256, 256)   # skip from enc3 (256ch)
        self.dec3 = DecoderBlock(256, 128, 128)   # skip from enc2 (128ch)
        self.dec2 = DecoderBlock(128, 64, 64)     # skip from enc1 (64ch)
        self.dec1 = DecoderBlock(64, 3, 32)       # skip from input (3ch)

        # ─── Output ──────────────────────────────────────────────────────
        self.output_conv = nn.Sequential(
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),  # Output in [0, 1] to match input range
        )

    def encode(self, x: torch.Tensor) -> tuple:
        """
        Encode input image, returning bottleneck features and skip connections.
        """
        e1 = self.enc1(x)    # (B, 64, 256, 256)
        e2 = self.enc2(e1)   # (B, 128, 128, 128)
        e3 = self.enc3(e2)   # (B, 256, 64, 64)
        e4 = self.enc4(e3)   # (B, 512, 32, 32)
        bn = self.bottleneck(e4)  # (B, 512, 32, 32)
        return bn, (x, e1, e2, e3)

    def decode(self, bn: torch.Tensor, skips: tuple) -> torch.Tensor:
        """
        Decode from bottleneck using skip connections for sharp reconstruction.
        """
        x_in, e1, e2, e3 = skips
        d4 = self.dec4(bn, e3)     # (B, 256, 64, 64)
        d3 = self.dec3(d4, e2)     # (B, 128, 128, 128)
        d2 = self.dec2(d3, e1)     # (B, 64, 256, 256)
        d1 = self.dec1(d2, x_in)   # (B, 32, 512, 512)
        out = self.output_conv(d1)  # (B, 3, 512, 512)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bn, skips = self.encode(x)
        return self.decode(bn, skips)


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick test: verify shapes and parameter count
    model = DocumentAutoencoder()
    print(f"Total parameters: {count_parameters(model):,}")

    x = torch.randn(1, 3, 512, 512)
    with torch.no_grad():
        out = model(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Output range: [{out.min():.3f}, {out.max():.3f}]")

    # Verify encoder output
    bn, skips = model.encode(x)
    print(f"Bottleneck shape: {bn.shape}")
    for i, s in enumerate(skips):
        print(f"Skip {i} shape: {s.shape}")
