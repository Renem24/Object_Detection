"""YOLOv1 model.
The original paper used Darknet‑19; here we implement a lighter
configurable CNN due to VRAM limits, while keeping (S=7, B=2, C=20).
"""

import torch
import torch.nn as nn


def _conv_block(in_channels, out_channels, kernel_size, stride):
    padding = 1 if kernel_size == 3 else 0
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.1),
    )


class Yolov1(nn.Module):
    def __init__(self, S=7, B=2, C=20):
        super().__init__()
        self.S, self.B, self.C = S, B, C

        # Backbone (similar to Tiny YOLO for speed)
        self.features = nn.Sequential(
            _conv_block(3, 16, 3, 1),
            nn.MaxPool2d(2, 2),
            _conv_block(16, 32, 3, 1),
            nn.MaxPool2d(2, 2),
            _conv_block(32, 64, 3, 1),
            nn.MaxPool2d(2, 2),
            _conv_block(64, 128, 3, 1),
            nn.MaxPool2d(2, 2),
            _conv_block(128, 256, 3, 1),
            nn.MaxPool2d(2, 2),
            _conv_block(256, 512, 3, 1),
            nn.MaxPool2d(2, 2),
            _conv_block(512, 1024, 3, 1),
            _conv_block(1024, 1024, 3, 1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * (self.S * self.S), 4096),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1),
            nn.Linear(4096, self.S * self.S * (self.C + self.B * 5)),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x.view(-1, self.S, self.S, self.C + self.B * 5)
