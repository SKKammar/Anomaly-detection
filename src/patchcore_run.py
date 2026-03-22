"""
patchcore_run.py - Run PatchCore end to end (no training needed).

Usage:
    python patchcore_run.py --category toothbrush --data_root "../data/mvtec"
"""

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from sklearn.metrics import roc_auc_score
from scipy.ndimage import gaussian_filter
import torch.nn.functional as F

import sys
sys.path.insert(0, ".")

from dataset import get_dataloaders
from models.patchcore import PatchCore


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, test_loader = get_dataloaders(
        data_root=args.data_root,
        category=args.category,
        img_size=args.img_size,
        batch_size=8,
    )

    # ── Build memory bank (replaces training) ─────────────────────────────────
    model = PatchCore(device=device)
    model.fit(train_loader)

    # ── Inference ─────────────────────────────────────────────────────────────
    all_image_scores = []
    all_image_labels = []
    all_pixel_scores  = []
    all_pixel_labels  = []

    heatmap_dir = Path(args.results_dir) / "heatmaps" / f"{args.category}_patchcore"
    heatmap_dir.mkdir(parents=True, exist_ok=True)

    print("Running inference on test set...")
    for i, (image, mask, label) in enumerate(test_loader):
        image = image.to(device)

        score_map, image_score = model.predict(image)

        # Upsample score map to img_size for pixel-level evaluation
        score_map_tensor = torch.tensor(score_map).unsqueeze(0).unsqueeze(0)
        score_map_up = F.interpolate(
            score_map_tensor,
            size=(args.img_size, args.img_size),
            mode="bilinear",
            align_corners=False
        ).squeeze().numpy()

        score_map_smooth = gaussian_filter(score_map_up, sigma=4)

        # Image-level score: mean of top-100 patches
        top_score = float(np.sort(score_map_smooth.flatten())[-100:].mean())
        all_image_scores.append(top_score)
        all_image_labels.append(label.item())

        mask_np = mask.squeeze().numpy()
        all_pixel_scores.append(score_map_smooth.flatten())
        all_pixel_labels.append(mask_np.flatten())

        # Save heatmaps
        if i < args.max_vis:
            img_np = image.squeeze().cpu().permute(1, 2, 0).clamp(0, 1).numpy()
            s_min, s_max = score_map_smooth.min(), score_map_smooth.max()
            score_norm = (score_map_smooth - s_min) / (s_max - s_min + 1e-8)

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            fig.suptitle(f"PatchCore | Label: {'Anomaly' if label.item() == 1 else 'Normal'}")

            axes[0].imshow(img_np)
            axes[0].set_title("Input image")
            axes[0].axis("off")

            heatmap_rgb = cm.jet(score_norm)[:, :, :3]
            overlay = 0.55 * img_np + 0.45 * heatmap_rgb
            axes[1].imshow(overlay.clip(0, 1))
            axes[1].set_title("Anomaly heatmap")
            axes[1].axis("off")

            axes[2].imshow(mask_np, cmap="gray", vmin=0, vmax=1)
            axes[2].set_title("Ground truth mask")
            axes[2].axis("off")

            plt.tight_layout()
            plt.savefig(heatmap_dir / f"sample_{i:03d}_label{label.item()}.png",
                        dpi=120, bbox_inches="tight")
            plt.close()

    # ── Metrics ───────────────────────────────────────────────────────────────
    img_auroc = roc_auc_score(all_image_labels, all_image_scores)

    pixel_scores = np.concatenate(all_pixel_scores)
    pixel_labels = np.concatenate(all_pixel_labels).astype(int)
    pix_auroc = roc_auc_score(pixel_labels, pixel_scores) if pixel_labels.sum() > 0 else 0.0

    print(f"\n{'='*45}")
    print(f"  Category:          {args.category}")
    print(f"  Method:            PatchCore")
    print(f"  Image-level AUROC: {img_auroc:.4f}")
    print(f"  Pixel-level AUROC: {pix_auroc:.4f}")
    print(f"  Heatmaps saved to: {heatmap_dir}")
    print(f"{'='*45}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--category",    type=str, default="toothbrush")
    parser.add_argument("--data_root",   type=str, default="../data/mvtec")
    parser.add_argument("--results_dir", type=str, default="../results")
    parser.add_argument("--img_size",    type=int, default=224)
    parser.add_argument("--max_vis",     type=int, default=20)
    args = parser.parse_args()
    run(args)