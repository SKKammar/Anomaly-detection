# InspectAI — Unsupervised Anomaly Detection in Industrial Inspection

A Computer Vision project that detects surface defects in industrial images **without any anomaly labels**.
Trained only on normal samples, the system learns what "normal" looks like and flags deviations at test time.

Two approaches are implemented and compared:
- **Convolutional Autoencoder** — reconstruction-based baseline
- **PatchCore** — pretrained feature memory bank (state of the art)

---

## Results on MVTec AD — Toothbrush Category

| Method | Image AUROC | Pixel AUROC | Training Time |
|--------|-------------|-------------|---------------|
| Convolutional Autoencoder | 0.36 | 0.83 | ~20 min (CPU) |
| **PatchCore** | **1.00** | **0.99** | **0 min** |

> PatchCore achieves perfect image-level classification and near-perfect pixel-level localization
> using only 60 normal training images and no GPU.

---

## Demo

A Flask web interface allows real-time inference — upload any toothbrush image and get
an anomaly heatmap back instantly.

![Demo](results/heatmaps/toothbrush_patchcore/sample_000_label1.png)

---

## How It Works

### Autoencoder (baseline)
```
Train on normal images only
      ↓
Model learns to reconstruct normal textures
      ↓
At test time: anomalous regions reconstruct poorly
      ↓
Reconstruction error per pixel = anomaly score
```

### PatchCore (state of the art)
```
Extract patch features from normal images (ResNet18 backbone)
      ↓
Store features in a memory bank (no training loop)
      ↓
At test time: find nearest neighbor distance per patch
      ↓
Distance = anomaly score → upsample → heatmap
```

Key insight: PatchCore never trains — it leverages features already learned by
ImageNet-pretrained ResNet18. The memory bank IS the model.

---

## Project Structure

```
Anomaly-detection/
├── src/
│   ├── models/
│   │   ├── autoencoder.py      # Conv autoencoder (encoder + decoder)
│   │   ├── patchcore.py        # PatchCore memory bank + inference
│   │   └── __init__.py
│   ├── templates/
│   │   └── index.html          # Web UI
│   ├── dataset.py              # MVTec AD dataloader
│   ├── train.py                # Autoencoder training loop
│   ├── evaluate.py             # AUROC metrics + heatmap generation
│   ├── patchcore_run.py        # PatchCore end-to-end (no training)
│   └── app.py                  # Flask web server
├── data/
│   └── mvtec/
│       └── toothbrush/         # MVTec AD category
├── checkpoints/                # Saved autoencoder weights
├── results/
│   └── heatmaps/               # Generated anomaly heatmap PNGs
└── requirements.txt
```

---

## Setup

### 1. Clone and create virtual environment

```bash
git clone <your-repo-url>
cd Anomaly-detection
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 2. Install PyTorch

```bash
# CPU only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# NVIDIA GPU (CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 3. Install remaining dependencies

```bash
pip install -r requirements.txt
```

### 4. Download MVTec AD dataset

Download from: https://www.mvtec.com/company/research/datasets/mvtec-ad

Extract into `data/mvtec/` so the structure looks like:
```
data/mvtec/toothbrush/
  train/good/          # 60 normal training images
  test/good/           # normal test images
  test/defective/      # anomalous test images
  ground_truth/        # pixel-level defect masks
```

---

## Usage

### Run PatchCore (recommended — no training needed)

```bash
cd src
python patchcore_run.py --category toothbrush --data_root "../data/mvtec"
```

Output:
```
Using device: cpu
[toothbrush] Train: 60 | Test: 42
Building memory bank from normal images...
Memory bank size: 47,040 patches
Running inference on test set...

=============================================
  Category:          toothbrush
  Method:            PatchCore
  Image-level AUROC: 1.0000
  Pixel-level AUROC: 0.9895
  Heatmaps saved to: ..\results\heatmaps\toothbrush_patchcore
=============================================
```

### Train and evaluate Autoencoder

```bash
cd src

# Train
python train.py --category toothbrush --epochs 300 --batch_size 8 --data_root "../data/mvtec" --lr 0.0002

# Evaluate
python evaluate.py --category toothbrush --data_root "../data/mvtec" --img_size 128
```

### Launch web interface

```bash
pip install flask
cd src
python app.py
```

Open `http://localhost:5000` in your browser. Upload any toothbrush image to get
a real-time anomaly heatmap.

---

## Dataset

**MVTec Anomaly Detection (MVTec AD)**
- 15 industrial categories (leather, bottle, cable, toothbrush, ...)
- ~3,600 normal training images + ~1,725 test images
- Pixel-level ground truth masks for all anomaly types
- Paper: [The MVTec Anomaly Detection Dataset (CVPR 2019)](https://openaccess.thecvf.com/content_CVPR_2019/html/Bergmann_MVTec_AD_--_A_Comprehensive_Real-World_Dataset_for_Unsupervised_Anomaly_CVPR_2019_paper.html)

---

## Key Concepts

**Why unsupervised?**
Collecting defect images in manufacturing is expensive and rare. Unsupervised methods
train on abundant normal images only — no defect labels needed.

**Why does PatchCore outperform Autoencoder?**
Autoencoders sometimes reconstruct defects well (generalize too much), making
reconstruction error an unreliable signal. PatchCore uses a frozen pretrained backbone —
features are already rich and discriminative. Comparing to a memory bank of normal
features gives a much sharper anomaly signal.

**AUROC explained**
- 0.50 = random guessing
- 0.75 = good baseline
- 0.90+ = strong result
- 1.00 = perfect (every anomaly ranked above every normal sample)

---

## Tech Stack

| Component | Library |
|-----------|---------|
| Deep learning | PyTorch 2.0 |
| Feature backbone | torchvision ResNet18 (ImageNet pretrained) |
| Dataset loading | torch.utils.data |
| Metrics | scikit-learn (roc_auc_score) |
| Visualization | matplotlib |
| Web interface | Flask |
| Image processing | Pillow, scipy |

---

## References

- **PatchCore**: Roth et al., *Towards Total Recall in Industrial Anomaly Detection* (CVPR 2022)
- **MVTec AD**: Bergmann et al., *MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection* (CVPR 2019)
- **Anomaly Detection Survey**: Pang et al., *Deep Learning for Anomaly Detection: A Review* (ACM Computing Surveys 2021)
