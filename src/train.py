"""
train.py - Train the Autoencoder on normal images only.

Usage:
    python src/train.py --category leather --epochs 100

The model learns to reconstruct normal textures/objects.
It will fail to reconstruct anomalous regions well → high residual = defect.
"""

import argparse
import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset import get_dataloaders
from models.autoencoder import build_model


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, _ = get_dataloaders(
        data_root=args.data_root,
        category=args.category,
        img_size=args.img_size,
        batch_size=args.batch_size,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model().to(device)

    # ── Loss: MSE + SSIM-style smoothness term ────────────────────────────────
    mse_loss = nn.MSELoss()

    # ── Optimizer + Scheduler ─────────────────────────────────────────────────
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_loss = float("inf")
    os.makedirs(args.ckpt_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0

        for batch_idx, (images, _, _) in enumerate(train_loader):
            images = images.to(device)

            # Forward pass
            reconstructed = model(images)
            loss = mse_loss(reconstructed, images)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        scheduler.step()

        # Logging
        if epoch % 10 == 0 or epoch == 1:
            lr = scheduler.get_last_lr()[0]
            print(f"Epoch [{epoch:3d}/{args.epochs}]  Loss: {avg_loss:.6f}  LR: {lr:.6f}")

        # Save best checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt_path = os.path.join(args.ckpt_dir, f"{args.category}_best.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": best_loss,
                "args": vars(args),
            }, ckpt_path)

    print(f"\nTraining complete. Best loss: {best_loss:.6f}")
    print(f"Checkpoint saved to: {ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Autoencoder for Anomaly Detection")
    parser.add_argument("--category",   type=str,   default="leather",
                        help="MVTec category (leather, bottle, cable, ...)")
    parser.add_argument("--data_root",  type=str,   default="data/mvtec")
    parser.add_argument("--ckpt_dir",   type=str,   default="checkpoints")
    parser.add_argument("--img_size",   type=int,   default=256)
    parser.add_argument("--batch_size", type=int,   default=16)
    parser.add_argument("--epochs",     type=int,   default=100)
    parser.add_argument("--lr",         type=float, default=1e-4)
    args = parser.parse_args()
    train(args)