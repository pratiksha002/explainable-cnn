import torch
import torch.nn as nn
from src.config import NUM_CLASSES, IMAGE_SIZE


# ─── Building Block 

class ConvBlock(nn.Module):

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        pool_size:    int   = 2,
        dropout_p:    float = 0.25
    ):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=pool_size, stride=pool_size),
            nn.Dropout2d(p=dropout_p)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ─── Full Model 

class IntelCNN(nn.Module):

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()

        # ── Convolutional Feature Extractor ──
        self.features = nn.Sequential(
            ConvBlock(in_channels=3,   out_channels=32,  dropout_p=0.25),
            ConvBlock(in_channels=32,  out_channels=64,  dropout_p=0.25),
            ConvBlock(in_channels=64,  out_channels=128, dropout_p=0.25),
        )

        # ── Global Average Pooling ──
        # Collapses (B, 128, H, W) → (B, 128) regardless of spatial size.
        # Advantage over Flatten: fewer params + better GradCAM compatibility.
        self.gap = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        # ── Classification Head ──
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes)     # Raw logits — no Softmax
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)    # Convolutional blocks
        x = self.gap(x)         # Global average pooling
        x = self.classifier(x)  # FC head → logits
        return x


# ─── Model Factory 
def build_model(device: torch.device) -> IntelCNN:
    model = IntelCNN(num_classes=NUM_CLASSES)
    model = model.to(device)
    return model


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)