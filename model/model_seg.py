import torch
import torch.nn as nn
import timm
import torch.nn.functional as F


class CBAM(nn.Module):
    def __init__(self, ch, r=16):
        super().__init__()

        # Channel attention
        self.mlp = nn.Sequential(
            nn.Linear(ch, ch // r),
            nn.ReLU(inplace=True),
            nn.Linear(ch // r, ch)
        )

        # Spatial attention
        self.spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3)

    def forward(self, x):
        b, c, h, w = x.shape

        # Channel attention
        avg = torch.mean(x, dim=(2, 3))
        mx = torch.max(x, dim=2)[0].max(dim=2)[0]
        ca = torch.sigmoid(self.mlp(avg) + self.mlp(mx)).view(b, c, 1, 1)
        x = x * ca

        # Spatial attention
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        sa = torch.sigmoid(self.spatial(torch.cat([avg, mx], dim=1)))
        return x * sa


class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Conv2d(F_g, F_int, 1)
        self.W_x = nn.Conv2d(F_l, F_int, 1)
        self.psi = nn.Conv2d(F_int, 1, 1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, g, x):
        if g.shape[2:] != x.shape[2:]:
            x = F.interpolate(
                x,
                size=g.shape[2:],
                mode="bilinear",
                align_corners=False
            )

        psi = self.relu(self.W_g(g) + self.W_x(x))
        psi = self.sigmoid(self.psi(psi))
        return x * psi

class AttDecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.att = AttentionGate(out_ch, skip_ch, out_ch // 2)

        self.conv = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            CBAM(out_ch),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.up(x)
        skip = self.att(x, skip)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class AttUNetDecoder(nn.Module):
    def __init__(self):
        super().__init__()

        # ConvNeXt-Tiny feature channels
        self.dec1 = AttDecoderBlock(768, 384, 256)  # 8→16
        self.dec2 = AttDecoderBlock(256, 192, 128)  # 16→32
        self.dec3 = AttDecoderBlock(128, 96, 64)    # 32→64

        # Final upsample to original resolution (64→256)
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4, stride=4),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

    def forward(self, feats):
        x = feats[3]      # [B, 768, 8, 8]
        x = self.dec1(x, feats[2])  # → [B, 256, 16, 16]
        x = self.dec2(x, feats[1])  # → [B, 128, 32, 32]
        x = self.dec3(x, feats[0])  # → [B, 64, 64, 64]
        x = self.final_up(x)        # → [B, 64, 256, 256]
        return x

class DualHeadAttConvNeXtUNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = timm.create_model(
            "convnext_tiny",
            pretrained=True,
            features_only=True,
            in_chans=3
        )

        self.cyt_decoder = AttUNetDecoder()
        self.nuc_decoder = AttUNetDecoder()

        self.cyt_head = nn.Conv2d(64, 1, kernel_size=1)
        self.nuc_head = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        feats = self.encoder(x)
        return {
            "cyt": self.cyt_head(self.cyt_decoder(feats)),
            "nuc": self.nuc_head(self.nuc_decoder(feats)),
        }

device = "cuda" if torch.cuda.is_available() else "cpu"
model = DualHeadAttConvNeXtUNet().to(device)
model.eval()

x = torch.randn(2, 3, 1024, 1024).to(device)

with torch.no_grad():
    out = model(x)

print("Input:", x.shape)
print("Cyt output:", out["cyt"].shape)
print("Nuc output:", out["nuc"].shape)
