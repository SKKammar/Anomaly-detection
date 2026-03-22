import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


class MVTecDataset(Dataset):
    def __init__(self, root, category, split="train", img_size=256):
        self.root = Path(root) / category
        self.split = split

        # No ImageNet normalization — keep images in [0, 1] to match sigmoid output
        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
        ])

        self.mask_transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
        ])

        self.img_size = img_size
        self.samples  = []
        self._load_samples()

    def _load_samples(self):
        if self.split == "train":
            good_dir = self.root / "train" / "good"
            for p in sorted(good_dir.glob("*.png")):
                self.samples.append((p, None, 0))
        else:
            for p in sorted((self.root / "test" / "good").glob("*.png")):
                self.samples.append((p, None, 0))

            for defect_dir in sorted((self.root / "test").iterdir()):
                if defect_dir.name == "good":
                    continue
                gt_dir = self.root / "ground_truth" / defect_dir.name
                for p in sorted(defect_dir.glob("*.png")):
                    mask_p = gt_dir / (p.stem + "_mask.png")
                    self.samples.append((p, mask_p if mask_p.exists() else None, 1))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path, label = self.samples[idx]

        image = self.transform(Image.open(img_path).convert("RGB"))

        if mask_path:
            mask = self.mask_transform(Image.open(mask_path).convert("L"))
        else:
            mask = torch.zeros(1, self.img_size, self.img_size)

        return image, mask, label


def get_dataloaders(data_root, category, img_size=256, batch_size=16):
    train_ds = MVTecDataset(data_root, category, split="train", img_size=img_size)
    test_ds  = MVTecDataset(data_root, category, split="test",  img_size=img_size)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=False)
    test_loader  = DataLoader(test_ds,  batch_size=1,          shuffle=False, num_workers=2, pin_memory=False)

    print(f"[{category}] Train: {len(train_ds)} | Test: {len(test_ds)}")
    return train_loader, test_loader