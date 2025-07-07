"""
Main file for training YOLOv1 on the Pascal VOC dataset
"""
import argparse
from pathlib import Path

import torch
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.transforms.functional as FT  # 아직 다른 모듈에서 사용될 수 있어 유지
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import Yolov1
from dataset import VOCDataset
from loss import YoloLoss
from utils import (
    non_max_suppression,
    mean_average_precision,
    intersection_over_union,
    cellboxes_to_boxes,
    get_bboxes,
    plot_image,
    save_checkpoint,
    load_checkpoint,
)

# ───────────────────────────────────────────────────────────
# 1) 전처리
# ───────────────────────────────────────────────────────────
class Compose:
    """간단한 transform 콤포저"""

    def __init__(self, transforms_):
        self.transforms = transforms_

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes
        return img, bboxes

# ───────────────────────────────────────────────────────────
# 2) 학습 루프
# ───────────────────────────────────────────────────────────
def train_fn(train_loader, model, optimizer, loss_fn, device):
    """one epoch of training"""
    model.train()  # 학습 모드 고정
    loop = tqdm(train_loader, leave=False)
    mean_loss = []

    for x, y in loop:
        x, y = x.to(device), y.to(device)

        out = model(x)
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_postfix(loss=loss.item())

    return sum(mean_loss) / len(mean_loss)

# ───────────────────────────────────────────────────────────
# 3) 평가 함수
# ───────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(loader, model, device):
    """mAP 계산 전용 (gradient 비활성)"""
    model.eval()  # ★ 평가 모드 고정

    pred_boxes, target_boxes = get_bboxes(
        loader, model, iou_threshold=0.5, threshold=0.4
    )

    mAP = mean_average_precision(
        pred_boxes,
        target_boxes,
        iou_threshold=0.5,
        box_format="midpoint",
    )
    return mAP

# ───────────────────────────────────────────────────────────
# 4) 메인
# ───────────────────────────────────────────────────────────
def main(args):
    torch.manual_seed(args.seed)  # (1) 시드 고정
    device = args.device
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # (2) 모델·옵티마이저·손실
    model = Yolov1(split_size=7, num_boxes=2, num_classes=20).to(device)
    optimizer = optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    loss_fn = YoloLoss()

    if args.load_model:
        load_checkpoint(
            torch.load(args.load_model_file, map_location=device), model, optimizer
        )

    # (3) 전처리 정의
    transform = Compose(
        [
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
        ]
    )

    # (4) 데이터셋·로더
    # use_csv 플래그에 따른 분기
    train_csv_path = args.train_csv if args.use_csv else None
    val_csv_path   = args.val_csv   if args.use_csv else None

    train_dataset = VOCDataset(
        csv_file=train_csv_path,
        transform=transform,
        img_dir=args.img_dir,
        label_dir=args.label_dir,
    )
    val_dataset = VOCDataset(
        csv_file=val_csv_path,
        transform=transform,
        img_dir=args.img_dir,
        label_dir=args.label_dir,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        shuffle=False,
        drop_last=False,
    )

    # (5) 학습/평가 사이클
    # best_map = 0.0  # ← 원본 코드에서 미사용. 필요시 주석 해제
    for epoch in range(args.epochs):
        avg_train_loss = train_fn(train_loader, model, optimizer, loss_fn, device)

        if epoch % 10 == 0 :
            print(
                f"[Epoch {epoch+1:03d}/{args.epochs}] "
                f"train_loss={avg_train_loss:.4f}"
            )

        if epoch % args.eval_interval == 0 or epoch == args.epochs - 1:
            mean_avg_prec = evaluate(val_loader, model, device)
            print(
                f"[Epoch {epoch+1:03d}/{args.epochs}] "
                f"train_loss={avg_train_loss:.4f}  mAP={mean_avg_prec:.4f}"
            )

            # # ----- 체크포인트 -----
            # if mean_avg_prec > best_map:
            #     best_map = mean_avg_prec
            #     checkpoint = {
            #         "state_dict": model.state_dict(),
            #         "optimizer": optimizer.state_dict(),
            #     }
            #     best_path = Path(args.checkpoint_dir) / args.best_model_name
            #     save_checkpoint(checkpoint, filename=str(best_path))
            #     print(f"✔️  New best model saved (mAP={best_map:.4f})")

    # (6) 마지막 체크포인트 저장
    print("Training complete. Saving final model weights...")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    final_path = Path(args.checkpoint_dir) / args.final_model_name
    save_checkpoint(checkpoint, filename=str(final_path))
    print(f"ℹ️  Final epoch weight saved → {final_path}")

# ───────────────────────────────────────────────────────────
# 5) argparse
# ───────────────────────────────────────────────────────────
def parse_args():
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    p = argparse.ArgumentParser("YOLOv1 Pascal-VOC 학습 스크립트")

    # 기본 설정
    p.add_argument("--seed",            type=int,   default=123)
    p.add_argument("--device",          choices=["cpu", "cuda"], default=default_device)

    # 하이퍼파라미터
    p.add_argument("--learning_rate",   type=float, default=2e-5)
    p.add_argument("--weight_decay",    type=float, default=0.0)
    p.add_argument("--batch_size",      type=int,   default=64)
    p.add_argument("--epochs",          type=int,   default=1000)
    p.add_argument("--eval_interval",   type=int,   default=30,
                   help="mAP 평가 간격(에폭 기준)")

    # DataLoader
    p.add_argument("--num_workers",     type=int,   default=2)
    p.add_argument("--pin_memory",      dest="pin_memory", action="store_true")
    p.add_argument("--no_pin_memory",   dest="pin_memory", action="store_false")
    p.set_defaults(pin_memory=True)

    # 경로
    p.add_argument("--img_size",        type=int,   default=448)
    p.add_argument("--img_dir",         default="data/images")
    p.add_argument("--label_dir",       default="data/labels")
    p.add_argument("--train_csv",       default="data/100examples.csv")
    p.add_argument("--val_csv",         default="data/test.csv")
    p.add_argument("--checkpoint_dir",  default="checkpoints")
    p.add_argument("--best_model_name", default="best.pth")
    p.add_argument("--final_model_name",default="final.pth")

    # 이어서 학습
    p.add_argument("--load_model",      action="store_true")
    p.add_argument("--load_model_file", default="overfit.pth.tar")

    # CSV 사용 여부 플래그 (기본값: False)
    p.add_argument("--use_csv", action="store_true",
                   help="CSV 파일(list) 기반 데이터셋 사용")

    return p.parse_args()

# ───────────────────────────────────────────────────────────
# 6) 실행 엔트리
# ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()
    main(args)
