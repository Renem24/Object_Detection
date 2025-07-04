"""
inference.py ─ YOLOv1 ▸ Pascal VOC ▸ 배치 이미지 추론
1) 아래 '설정' 블록 값만 바꾼 뒤
   python inference.py
"""

from pathlib import Path
from typing import List
import random

import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont

from model import Yolov1
from utils import (
    cellboxes_to_boxes,
    non_max_suppression,
)

# ───────────────────────────────────────────────────────────
# 1) 설정
# ───────────────────────────────────────────────────────────
DEVICE            = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE          = 448

# 경로
INPUT_DIR         = "data/example_images"
OUTPUT_DIR        = "data/inference_output"
WEIGHTS_PATH      = "checkpoints/best.pth"

# 추론 파라미터
IOU_THRESHOLD     = 0.5
CONF_THRESHOLD    = 0.4

# Pascal VOC 20 class names
CLASS_NAMES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]

# 클래스별 고정 색상 생성 (R,G,B)  ─ reproducible 하도록 시드 고정
random.seed(42)
COLOR_MAP = {
    idx: tuple(random.randint(64, 255) for _ in range(3))  # 64-255 범위로 부드러운 색
    for idx in range(len(CLASS_NAMES))
}

# ───────────────────────────────────────────────────────────
# 2) 전처리
# ───────────────────────────────────────────────────────────
class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes=None):
        for t in self.transforms:
            img, bboxes = t(img), bboxes
        return img, bboxes


transform = Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ]
)

# ───────────────────────────────────────────────────────────
# 3) 보조 함수
# ───────────────────────────────────────────────────────────
def draw_boxes_pil(image: Image.Image, boxes: List[List[float]]) -> Image.Image:
    """PIL 이미지 위에 예측 박스 + 클래스명/신뢰도 그리기(클래스별 고정 색)"""
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()          # 시스템 기본 비트맵 폰트
    w, h = image.size

    for cls, x, y, bw, bh, conf in boxes:
        if conf < CONF_THRESHOLD:
            continue

        # 좌표 변환 (midpoint → 왼쪽 위/오른쪽 아래)
        x1 = (x - bw / 2) * w
        y1 = (y - bh / 2) * h
        x2 = (x + bw / 2) * w
        y2 = (y + bh / 2) * h

        color = COLOR_MAP[int(cls)]
        label = f"{CLASS_NAMES[int(cls)]}: {conf:.2f}"

        # 바운딩 박스
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        # 텍스트 크기 계산 (Pillow 버전별 호환)
        if hasattr(draw, "textbbox"):                    # Pillow ≥ 8.0
            tb = draw.textbbox((x1, y1), label, font=font)
            text_w, text_h = tb[2] - tb[0], tb[3] - tb[1]
        else:                                            # Pillow < 8.0
            text_w, text_h = draw.textsize(label, font=font)

        # 라벨 배경 & 텍스트
        draw.rectangle([x1, y1 - text_h, x1 + text_w, y1], fill=color)
        draw.text((x1, y1 - text_h), label, fill="black", font=font)

    return image


@torch.no_grad()
def predict_image(image_path: Path, model: torch.nn.Module) -> Image.Image:
    """단일 이미지 추론 후 박스가 그려진 PIL 이미지 반환"""
    orig_img = Image.open(image_path).convert("RGB")
    img, _ = transform(orig_img)
    img = img.unsqueeze(0).to(DEVICE)

    model.eval()
    preds = model(img)
    bboxes = cellboxes_to_boxes(preds, S=7)
    bboxes = non_max_suppression(
        bboxes[0],
        iou_threshold=IOU_THRESHOLD,
        threshold=CONF_THRESHOLD,
        box_format="midpoint",
    )

    return draw_boxes_pil(orig_img, bboxes)

# ───────────────────────────────────────────────────────────
# 4) main
# ───────────────────────────────────────────────────────────
def main() -> None:
    model = Yolov1(split_size=7, num_boxes=2, num_classes=20).to(DEVICE)

    ckpt = Path(WEIGHTS_PATH)
    if not ckpt.exists():
        raise FileNotFoundError(ckpt)
    state = torch.load(ckpt, map_location=DEVICE)
    model.load_state_dict(state["state_dict"])
    print(f"✔️  Loaded weights: {WEIGHTS_PATH}")
    print(f"‣ Device         : {DEVICE}")

    inp_dir = Path(INPUT_DIR)
    out_dir = Path(OUTPUT_DIR)
    if not inp_dir.exists():
        raise FileNotFoundError(inp_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = sorted([p for p in inp_dir.iterdir()
                     if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}])
    if not images:
        raise ValueError("입력 폴더에 이미지가 없습니다.")

    for img_path in images:
        print(f"[Inference] {img_path.name}")
        result_img = predict_image(img_path, model)

        save_path = out_dir / f"{img_path.stem}_detect.jpg"
        result_img.save(save_path)
        print(f"   → Saved: {save_path}")

if __name__ == "__main__":
    main()
