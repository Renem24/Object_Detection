"""Pascal VOC dataset loader for YOLOv1.
Expects a CSV file where each row is [img_path,label_path].
label_path is a txt file with per‑line: class x y w h (all normalized 0‑1).
"""
from pathlib import Path
import os
from typing import List, Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


class Compose:
    """Compose both image and bbox transforms so they stay aligned."""

    def __init__(self, img_size: int = 448):
        self.transforms = T.Compose(
            [
                T.Resize((img_size, img_size)),
                T.ToTensor(),
            ]
        )
        self.img_size = img_size

    def __call__(self, img: Image.Image, bboxes: List[List[float]]):
        return self.transforms(img), torch.tensor(bboxes)


class VOCDataset(Dataset):
    def __init__(
        self,
        csv_file: str | None = None,
        img_dir: str | None = None,
        label_dir: str | None = None,
        S: int = 7,
        B: int = 2,
        C: int = 20,
        img_size: int = 448,
    ):
        if csv_file is not None:
            # ── CSV 파일이 주어진 경우 ───────────────────────────────
            self.annotations = pd.read_csv(csv_file)
        else:
            # ── CSV 없이 폴더를 직접 스캔하는 경우 ──────────────────
            assert img_dir and label_dir, (
                "csv_file을 주지 않을 때는 img_dir과 label_dir 둘 다 필요합니다."
            )
            names = sorted(os.listdir(img_dir))
            self.annotations = pd.DataFrame({
                0: [os.path.join(img_dir, n) for n in names],
                1: [os.path.join(label_dir,
                                    Path(n).with_suffix('.txt').name) for n in names],
            })

        self.S, self.B, self.C = S, B, C
        self.transform = Compose(img_size)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, label_path = self.annotations.iloc[index]
        image = Image.open(img_path).convert("RGB")

        boxes = []
        with open(label_path) as f:
            for line in f.readlines():
                cls, x, y, w, h = map(float, line.split())
                boxes.append([int(cls), x, y, w, h])

        image, boxes = self.transform(image, boxes)
        label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B))

        for box in boxes:
            cls, x, y, w, h = box
            i, j = int(self.S * y), int(self.S * x)
            x_cell, y_cell = self.S * x - j, self.S * y - i
            w_cell, h_cell = w * self.S, h * self.S
            if label_matrix[i, j, self.C] == 0:
                label_matrix[i, j, self.C] = 1  # objectness
                label_matrix[i, j, self.C + 1 : self.C + 5] = torch.tensor([x_cell, y_cell, w_cell, h_cell])
                label_matrix[i, j, int(cls)] = 1

        return image, label_matrix
