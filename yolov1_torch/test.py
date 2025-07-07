"""
test.py  ─ YOLOv1  ▸  Pascal VOC  ▸  mAP 평가 스크립트
(argparse 적용 버전)
"""
import argparse
import os
import time
from pathlib import Path

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
# 1) 전처리
# ───────────────────────────────────────────────────────────
class Compose:
    def __init__(self, transforms_):
        self.transforms = transforms_

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes
        return img, bboxes


# ───────────────────────────────────────────────────────────
# 2) mAP 계산
# ───────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(loader, model, iou_thr, score_thr):
    model.eval()
    pred_boxes, target_boxes = get_bboxes(
        loader,
        model,
        iou_threshold=iou_thr,
        threshold=score_thr,
    )
    mAP = mean_average_precision(
        pred_boxes,
        target_boxes,
        iou_threshold=iou_thr,
        box_format="midpoint",
    )
    return mAP


# ───────────────────────────────────────────────────────────
# 3) 메인
# ───────────────────────────────────────────────────────────
def main(args) -> None:
    print("‣ Device :", args.device)

    # ----- 모델 & 체크포인트 -----
    model = Yolov1(split_size=7, num_boxes=2, num_classes=20).to(args.device)
    if not os.path.exists(args.best_model_path):
        raise FileNotFoundError(f"{args.best_model_path} not found.")
    load_checkpoint(torch.load(args.best_model_path, map_location=args.device), model)
    print(f"✔️  Loaded weights from {args.best_model_path}")

    # ----- 전처리 -----
    transform = Compose(
        [
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
        ]
    )

    # ----- DataLoader -----
    test_dataset = VOCDataset(
        csv_file=args.test_csv,
        transform=transform,
        img_dir=args.img_dir,
        label_dir=args.label_dir,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        shuffle=False,
        drop_last=False,
    )

    # ----- 평가 -----
    start = time.perf_counter()
    mAP = evaluate(test_loader, model, args.iou_threshold, args.score_threshold)
    elapsed = time.perf_counter() - start

    print(f"\n================  TEST RESULT  ================\n"
          f"mAP@{args.iou_threshold:.2f} = {mAP:.4f}\n"
          f"(elapsed {elapsed:.1f} s, "
          f"images {len(test_dataset)}, "
          f"batch size {args.batch_size})\n"
          f"===============================================\n")


# ───────────────────────────────────────────────────────────
# 4) argparse
# ───────────────────────────────────────────────────────────
def parse_args():
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    p = argparse.ArgumentParser("YOLOv1 Pascal-VOC mAP 평가 스크립트")

    # 기본 실행 옵션
    p.add_argument("--device", choices=["cpu", "cuda"], default=default_device)
    p.add_argument("--batch_size",      type=int,   default=16)
    p.add_argument("--num_workers",     type=int,   default=2)
    p.add_argument("--pin_memory",      dest="pin_memory", action="store_true")
    p.add_argument("--no_pin_memory",   dest="pin_memory", action="store_false")
    p.set_defaults(pin_memory=torch.cuda.is_available())

    # 경로
    p.add_argument("--best_model_path", default="checkpoints/best.pth")
    p.add_argument("--img_dir",         default="data/images")
    p.add_argument("--label_dir",       default="data/labels")
    p.add_argument("--test_csv",        default="data/test.csv")

    # 평가 파라미터
    p.add_argument("--iou_threshold",   type=float, default=0.5)
    p.add_argument("--score_threshold", type=float, default=0.4)
    p.add_argument("--img_size",        type=int,   default=448)

    return p.parse_args()


# ───────────────────────────────────────────────────────────
# 5) 실행 엔트리
# ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()
    main(args)
