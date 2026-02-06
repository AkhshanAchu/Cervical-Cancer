import torch
import torch.nn as nn
import timm
import torch.nn.functional as F
from torchvision.models import convnext_tiny


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class FeatureSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # x: [B, D]
        x = x.unsqueeze(1)               # [B, 1, D]
        attn_out, _ = self.attn(x, x, x)
        x = self.norm(x + attn_out)
        return x.squeeze(1)

class AttentionMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()

        self.fc1 = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4)
        )

        self.attn = FeatureSelfAttention(512, num_heads=4)

        self.fc2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.fc1(x)
        x = self.attn(x)
        x = self.fc2(x)
        return x


class ConvNeXtAttentionClassifier(nn.Module):
    def __init__(self, num_classes=5, input_channels=5):
        super().__init__()

        self.backbone = convnext_tiny(pretrained=True)
        self.backbone.features[0][0] = nn.Conv2d(
            input_channels, 96, kernel_size=4, stride=4
        )

        self.se = SEBlock(768)

        self.classifier = AttentionMLP(768,num_classes)

    def forward(self, x):
        x = self.backbone.features(x)    # [B, 768, H, W]
        x = self.se(x)                   # Channel attention
        x = x.mean(dim=[2, 3])           # GAP
        x = self.classifier(x)
        return x

model = ConvNeXtAttentionClassifier(
    num_classes=5,
    input_channels=5
)

print(model(torch.randn(2,5,256,256)).shape)