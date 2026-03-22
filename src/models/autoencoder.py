import torch
import torch.nn as nn


class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, last=False):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Identity() if last else nn.BatchNorm2d(out_ch),
            nn.Sigmoid()  if last else nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ConvAutoencoder(nn.Module):
    """
    Input images must be in [0, 1] range (use ToTensor() only, no normalization).
    Sigmoid output matches [0, 1] input — MSE loss works correctly.
    """
    def __init__(self, img_channels=3, base_ch=32):
        super().__init__()

        self.encoder = nn.Sequential(
            EncoderBlock(img_channels, base_ch),       # 3   -> 32
            EncoderBlock(base_ch,     base_ch * 2),    # 32  -> 64
            EncoderBlock(base_ch * 2, base_ch * 4),    # 64  -> 128
            EncoderBlock(base_ch * 4, base_ch * 8),    # 128 -> 256
        )

        self.decoder = nn.Sequential(
            DecoderBlock(base_ch * 8, base_ch * 4),              # 256 -> 128
            DecoderBlock(base_ch * 4, base_ch * 2),              # 128 -> 64
            DecoderBlock(base_ch * 2, base_ch),                  # 64  -> 32
            DecoderBlock(base_ch,     img_channels, last=True),  # 32  -> 3
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def anomaly_score(self, x):
        with torch.no_grad():
            x_hat = self.forward(x)
            residual   = (x - x_hat) ** 2
            score_map  = residual.mean(dim=1, keepdim=True)
            image_score = score_map.flatten(1).max(dim=1).values
        return score_map, image_score


def build_model(img_channels=3, base_ch=32):
    model = ConvAutoencoder(img_channels=img_channels, base_ch=base_ch)
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total:,}")
    return model


if __name__ == "__main__":
    model = build_model()
    x = torch.rand(2, 3, 256, 256)   # [0,1] range
    out = model(x)
    print(f"Input: {x.shape} -> Output: {out.shape}")
    print(f"Output range: [{out.min():.3f}, {out.max():.3f}]")  # should be ~[0,1]