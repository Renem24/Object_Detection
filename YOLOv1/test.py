"""
test.py  ─ YOLOv1  ▸  Pascal VOC  ▸  mAP 평가 스크립트
"""

import os
import time
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from model import Yolov1
from dataset import VOCDataset
from utils import (
    mean_average_precision,
    get_bboxes,
    load_checkpoint,
)

# ───────────────────────────────────────────────────────────
# 1) 설정
# ───────────────────────────────────────────────────────────
DEVICE            = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE        = 16
NUM_WORKERS       = 2
PIN_MEMORY        = torch.cuda.is_available()
BEST_MODEL_PATH   = "checkpoints/best.pth"         # ★ 경로 수정

IMG_DIR           = "data/images"
LABEL_DIR         = "data/labels"
TEST_CSV          = "data/test.csv"

IOU_THRESHOLD     = 0.5
SCORE_THRESHOLD   = 0.4

# ───────────────────────────────────────────────────────────
# 2) 데이터 전처리
# ───────────────────────────────────────────────────────────
class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes
        return img, bboxes


transform = Compose(
    [
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
    ]
)

# ───────────────────────────────────────────────────────────
# 3) mAP 계산
# ───────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(loader, model):
    model.eval()

    pred_boxes, target_boxes = get_bboxes(
        loader,
        model,
        iou_threshold=IOU_THRESHOLD,
        threshold=SCORE_THRESHOLD,
    )

    mAP = mean_average_precision(
        pred_boxes,
        target_boxes,
        iou_threshold=IOU_THRESHOLD,
        box_format="midpoint",
    )
    return mAP

# ───────────────────────────────────────────────────────────
# 4) 메인
# ───────────────────────────────────────────────────────────
def main() -> None:
    print("‣ Device :", DEVICE)

    # ----- 모델 & 체크포인트 -----
    model = Yolov1(split_size=7, num_boxes=2, num_classes=20).to(DEVICE)
    if not os.path.exists(BEST_MODEL_PATH):
        raise FileNotFoundError(f"{BEST_MODEL_PATH} not found.")
    load_checkpoint(torch.load(BEST_MODEL_PATH, map_location=DEVICE), model)
    print(f"✔️  Loaded weights from {BEST_MODEL_PATH}")

    # ----- DataLoader -----
    test_dataset = VOCDataset(
        csv_file=TEST_CSV,
        transform=transform,
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=False,
        drop_last=False,
    )

    # ----- 평가 -----
    start = time.perf_counter()
    mAP = evaluate(test_loader, model)
    elapsed = time.perf_counter() - start

    print(f"\n================  TEST RESULT  ================\n"
          f"mAP@{IOU_THRESHOLD:.2f} = {mAP:.4f}\n"
          f"(elapsed {elapsed:.1f} s, "
          f"images {len(test_dataset)}, "
          f"batch size {BATCH_SIZE})\n"
          f"===============================================\n")

if __name__ == "__main__":
    main()
