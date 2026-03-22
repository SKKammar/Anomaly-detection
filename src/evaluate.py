"""
evaluate.py - Evaluate trained model and generate anomaly heatmaps.

Computes:
  - Image-level AUROC  (is the whole image anomalous?)
  - Pixel-level AUROC  (which pixels are anomalous?)
  - Saves heatmap overlays to results/heatmaps/

Usage:
    python src/evaluate.py --category leather
"""

import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from sklearn.metrics import roc_auc_score
from scipy.ndimage import gaussian_filter

from dataset import get_dataloaders
from models.autoencoder import build_model


def load_model(ckpt_path: str, device: torch.device):
    model = build_model().to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt['epoch']} (loss={ckpt['loss']:.6f})")
    return model


def denormalize(tensor):
    return tensor.cpu().clamp(0, 1)

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load model ─────────────────────────────────────────────────────────────
    ckpt_path = os.path.join(args.ckpt_dir, f"{args.category}_best.pth")
    model = load_model(ckpt_path, device)

    # ── Data ───────────────────────────────────────────────────────────────────
    _, test_loader = get_dataloaders(
        data_root=args.data_root,
        category=args.category,
        img_size=args.img_size,
        batch_size=1,
    )

    heatmap_dir = Path(args.results_dir) / "heatmaps" / args.category
    heatmap_dir.mkdir(parents=True, exist_ok=True)

    # ── Inference ──────────────────────────────────────────────────────────────
    all_image_scores = []
    all_image_labels = []
    all_pixel_scores = []
    all_pixel_labels = []

    for i, (image, mask, label) in enumerate(test_loader):
        image = image.to(device)

        score_map, image_score = model.anomaly_score(image)

        # Upsample score map to original image size
        score_map_np = score_map.squeeze().cpu().numpy()           # (H, W)
        score_map_smooth = gaussian_filter(score_map_np, sigma=8)  # smooth for better localization

        # top-100 pixel average instead of single max — more stable image score
        score_map_flat = torch.tensor(score_map_smooth).flatten()
        top100 = score_map_flat.topk(100).values.mean().item()
        all_image_scores.append(top100)
        all_image_labels.append(label.item())

        mask_np = mask.squeeze().numpy()  # (H, W) — 0/1 ground truth
        all_pixel_scores.append(score_map_smooth.flatten())
        all_pixel_labels.append(mask_np.flatten())

        # Save heatmap visualizations (every sample or limit to first N)
        if i < args.max_vis:
            save_heatmap(
                image=image.squeeze(),
                score_map=score_map_smooth,
                gt_mask=mask_np,
                label=label.item(),
                save_path=heatmap_dir / f"sample_{i:03d}_label{label.item()}.png",
            )

    # ── Metrics ────────────────────────────────────────────────────────────────
    img_auroc = roc_auc_score(all_image_labels, all_image_scores)

    pixel_scores = np.concatenate(all_pixel_scores)
    pixel_labels = np.concatenate(all_pixel_labels).astype(int)
    pix_auroc = roc_auc_score(pixel_labels, pixel_scores) if pixel_labels.sum() > 0 else 0.0

    print(f"\n{'='*45}")
    print(f"  Category:          {args.category}")
    print(f"  Image-level AUROC: {img_auroc:.4f}")
    print(f"  Pixel-level AUROC: {pix_auroc:.4f}")
    print(f"  Heatmaps saved to: {heatmap_dir}")
    print(f"{'='*45}\n")

    return img_auroc, pix_auroc


def save_heatmap(image, score_map, gt_mask, label, save_path):
    """
    Save a 4-panel figure:
      1. Original image
      2. Reconstruction (what the AE outputs)
      3. Anomaly heatmap (score map overlaid)
      4. Ground truth mask
    """
    img_np = denormalize(image).permute(1, 2, 0).numpy()

    # Normalize score map to [0, 1] for visualization
    s_min, s_max = score_map.min(), score_map.max()
    score_norm = (score_map - s_min) / (s_max - s_min + 1e-8)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(f"Label: {'Anomaly' if label == 1 else 'Normal'}", fontsize=13)

    axes[0].imshow(img_np)
    axes[0].set_title("Input image")
    axes[0].axis("off")

    # Heatmap overlay: blend original image with red channel
    heatmap_rgb = cm.jet(score_norm)[:, :, :3]
    overlay = 0.55 * img_np + 0.45 * heatmap_rgb
    axes[1].imshow(overlay.clip(0, 1))
    axes[1].set_title("Anomaly heatmap")
    axes[1].axis("off")

    axes[2].imshow(gt_mask, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("Ground truth mask")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Anomaly Detection Model")
    parser.add_argument("--category",   type=str,  default="leather")
    parser.add_argument("--data_root",  type=str,  default="data/mvtec")
    parser.add_argument("--ckpt_dir",   type=str,  default="checkpoints")
    parser.add_argument("--results_dir",type=str,  default="results")
    parser.add_argument("--img_size",   type=int,  default=256)
    parser.add_argument("--max_vis",    type=int,  default=20,
                        help="Max number of heatmaps to save")
    args = parser.parse_args()
    evaluate(args)