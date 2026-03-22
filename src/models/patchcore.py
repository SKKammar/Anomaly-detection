import torch
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import DataLoader
import numpy as np


class PatchCore:
    """
    PatchCore - No training needed.

    Steps:
      1. Extract patch features from normal images using pretrained ResNet18
      2. Store in memory bank
      3. At test time, find nearest neighbor distance = anomaly score
    """

    def __init__(self, device, backbone="resnet18"):
        self.device = device
        self.memory_bank = None

        # Load pretrained ResNet18 — downloads ~45 MB automatically
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Use features from layer2 + layer3 (best for anomaly detection)
        self.layer2 = torch.nn.Sequential(*list(resnet.children())[:6])
        self.layer3 = torch.nn.Sequential(*list(resnet.children())[:7])

        self.layer2.to(device).eval()
        self.layer3.to(device).eval()

        for p in self.layer2.parameters():
            p.requires_grad = False
        for p in self.layer3.parameters():
            p.requires_grad = False

    def extract_features(self, x):
        """Extract and concatenate features from layer2 and layer3."""
        with torch.no_grad():
            f2 = self.layer2(x)   # (B, 128, H/8,  W/8)
            f3 = self.layer3(x)   # (B, 256, H/16, W/16)

            # Upsample f3 to match f2 spatial size
            f3 = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)

            # Concatenate along channel dim
            features = torch.cat([f2, f3], dim=1)  # (B, 384, H/8, W/8)
        return features

    def fit(self, train_loader):
        """Build memory bank from normal training images."""
        print("Building memory bank from normal images...")
        all_features = []

        for images, _, _ in train_loader:
            images = images.to(self.device)
            features = self.extract_features(images)  # (B, 384, h, w)

            # Reshape to (B*h*w, 384) — each patch is one row
            B, C, h, w = features.shape
            features = features.permute(0, 2, 3, 1).reshape(-1, C)
            all_features.append(features.cpu())

        self.memory_bank = torch.cat(all_features, dim=0)  # (N_patches, 384)
        print(f"Memory bank size: {self.memory_bank.shape[0]:,} patches")

    def predict(self, image):
        """
        Returns:
            score_map  : (h, w) numpy array — per-patch anomaly scores
            image_score: float — max patch score = image-level anomaly score
        """
        features = self.extract_features(image)   # (1, 384, h, w)
        B, C, h, w = features.shape

        # Reshape to (h*w, 384)
        patches = features.permute(0, 2, 3, 1).reshape(-1, C).cpu()

        # Compute distance from each patch to its nearest neighbor in memory bank
        # Using chunked computation to avoid OOM on large memory banks
        chunk_size = 1000
        distances = []
        for i in range(0, patches.shape[0], chunk_size):
            chunk = patches[i:i+chunk_size]                        # (chunk, 384)
            dists = torch.cdist(chunk, self.memory_bank)           # (chunk, N)
            min_dists = dists.min(dim=1).values                    # (chunk,)
            distances.append(min_dists)

        distances = torch.cat(distances)           # (h*w,)
        score_map  = distances.reshape(h, w).numpy()
        image_score = float(distances.max())

        return score_map, image_score