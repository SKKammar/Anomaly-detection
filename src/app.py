"""
app.py - Flask web interface for PatchCore anomaly detection.

Usage:
    cd src
    python app.py

Then open http://localhost:5000 in your browser.
"""

import io
import base64
import sys
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — required for Flask
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from flask import Flask, request, jsonify, render_template_string
from PIL import Image
from scipy.ndimage import gaussian_filter
import torchvision.transforms as T

sys.path.insert(0, ".")
from models.patchcore import PatchCore
from dataset import get_dataloaders

app = Flask(__name__)

# ── Global model (loaded once at startup) ─────────────────────────────────────
device = torch.device("cpu")
model  = None

IMG_SIZE = 224

def load_model():
    global model
    print("Loading PatchCore and building memory bank...")
    model = PatchCore(device=device)
    train_loader, _ = get_dataloaders(
        data_root="../data/mvtec",
        category="toothbrush",
        img_size=IMG_SIZE,
        batch_size=8,
    )
    model.fit(train_loader)
    print("Model ready.")

# ── Image transform (same as training) ────────────────────────────────────────
transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
])

# ── Helper: run inference and return base64 heatmap image ─────────────────────
def run_inference(pil_image):
    tensor = transform(pil_image.convert("RGB")).unsqueeze(0).to(device)

    score_map, image_score = model.predict(tensor)

    # Upsample score map to IMG_SIZE
    score_map_t = torch.tensor(score_map).unsqueeze(0).unsqueeze(0)
    score_map_up = F.interpolate(
        score_map_t, size=(IMG_SIZE, IMG_SIZE),
        mode="bilinear", align_corners=False
    ).squeeze().numpy()

    score_map_smooth = gaussian_filter(score_map_up, sigma=4)

    # Image-level score
    top_score = float(np.sort(score_map_smooth.flatten())[-100:].mean())

    # Normalize for visualization
    s_min, s_max = score_map_smooth.min(), score_map_smooth.max()
    score_norm = (score_map_smooth - s_min) / (s_max - s_min + 1e-8)

    # Original image as numpy
    img_np = np.array(pil_image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))) / 255.0

    # Heatmap overlay
    heatmap_rgb = cm.jet(score_norm)[:, :, :3]
    overlay = (0.55 * img_np + 0.45 * heatmap_rgb).clip(0, 1)

    # Build 1x2 figure: original | heatmap
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor="#0f0f0f")
    for ax in axes:
        ax.axis("off")
    axes[0].imshow(img_np)
    axes[0].set_title("Input", color="white", fontsize=13, pad=8)
    axes[1].imshow(overlay)
    axes[1].set_title("Anomaly heatmap", color="white", fontsize=13, pad=8)
    plt.tight_layout(pad=1.5)

    # Encode to base64
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor="#0f0f0f")
    plt.close()
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")

    # Determine verdict
    threshold = 2.082
    verdict = "ANOMALY DETECTED" if top_score > threshold else "NORMAL"
    confidence = min(100, int(top_score / 0.03 * 100)) if top_score > threshold else int((1 - top_score / threshold) * 100)

    return {
        "heatmap": img_b64,
        "score": round(float(top_score), 6),
        "verdict": verdict,
        "confidence": confidence,
    }


# ── Routes ────────────────────────────────────────────────────────────────────
HTML = open("templates/index.html").read()

@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    file = request.files["image"]
    try:
        pil_image = Image.open(file.stream)
        result = run_inference(pil_image)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    load_model()
    print("\nOpen http://localhost:5000 in your browser\n")
    app.run(debug=False, port=5000)