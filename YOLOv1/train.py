"""
Main file for training YOLOv1 on the Pascal VOC dataset
"""

import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as FT  # 아직 다른 모듈에서 사용될 수 있어 유지
from tqdm import tqdm
from torch.utils.data import DataLoader

from model import Yolov1
from dataset import VOCDataset
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
from loss import YoloLoss

# ───────────────────────────────────────────────────────────
# 1) 환경 설정
# ───────────────────────────────────────────────────────────
seed = 123
torch.manual_seed(seed)

LEARNING_RATE = 2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64          # 원 논문 64지만 VRAM 한계 고려
WEIGHT_DECAY = 0
EPOCHS = 1000
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = False
LOAD_MODEL_FILE = "overfit.pth.tar"
BEST_MODEL_PATH = "checkpoints/best.pth"
FINAL_MODEL_PATH = "checkpoints/final.pth"

IMG_DIR = "data/images"
LABEL_DIR = "data/labels"

# ───────────────────────────────────────────────────────────
# 2) 데이터 전처리
# ───────────────────────────────────────────────────────────
class Compose:
    """간단한 transform 콤포저"""

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
# 3) 학습 루프
# ───────────────────────────────────────────────────────────
def train_fn(train_loader, model, optimizer, loss_fn):
    """one epoch of training"""
    model.train()  # ★ 학습 모드 고정
    loop = tqdm(train_loader, leave=True)
    mean_loss = []

    for x, y in loop:
        x, y = x.to(DEVICE), y.to(DEVICE)

        out = model(x)
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_postfix(loss=loss.item())

    return sum(mean_loss) / len(mean_loss)


# ───────────────────────────────────────────────────────────
# 4) 평가 함수
# ───────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(loader, model):
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
# 5) 메인
# ───────────────────────────────────────────────────────────
def main():
    model = Yolov1(split_size=7, num_boxes=2, num_classes=20).to(DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    loss_fn = YoloLoss()

    if LOAD_MODEL:
        load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)

    # ----- DataLoader -----
    train_dataset = VOCDataset(
        csv_file="data/100examples.csv",
        transform=transform,
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR,
    )
    val_dataset = VOCDataset(  # 검증용 (원하시면 별도 CSV 사용)
        csv_file="data/test.csv",
        transform=transform,
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=False,
        drop_last=False,
    )

    # ----- 학습/평가 사이클 -----
    best_map = 0.0
    for epoch in range(EPOCHS):
        avg_train_loss = train_fn(train_loader, model, optimizer, loss_fn)

        if epoch % 10 == 0 or epoch == EPOCHS - 1:
            mean_avg_prec = evaluate(val_loader, model)
            print(
                f"[Epoch {epoch+1:03d}/{EPOCHS}] "
                f"train_loss={avg_train_loss:.4f}  mAP={mean_avg_prec:.4f}"
            )

            # ----- 체크포인트 -----
            if mean_avg_prec > best_map:
                best_map = mean_avg_prec
                checkpoint = {
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                save_checkpoint(checkpoint, filename=BEST_MODEL_PATH)
                print(f"✔️  New best model saved (mAP={best_map:.4f})")
            
    
    save_checkpoint(checkpoint, filename=FINAL_MODEL_PATH)
    print(f"ℹ️  Final epoch weight saved → {FINAL_MODEL_PATH}")


if __name__ == "__main__":
    main()
