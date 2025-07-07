"""
inference.py ─ YOLOv1 ▸ Pascal VOC ▸ 배치 이미지 추론
CLI 예:
    python inference.py --input_dir data/my_imgs --output_dir runs
"""
from pathlib import Path
from typing import List
import random
import argparse

import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont

from model import Yolov1
from utils import cellboxes_to_boxes, non_max_suppression


# ───────────────────────────────────────────────────────────
# 1) 클래스 이름·컬러맵 (전역에 그대로 유지)
# ───────────────────────────────────────────────────────────
CLASS_NAMES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]

random.seed(42)
COLOR_MAP = {
    idx: tuple(random.randint(64, 255) for _ in range(3))
    for idx in range(len(CLASS_NAMES))
}


# ───────────────────────────────────────────────────────────
# 2) 전처리용 Compose
# ───────────────────────────────────────────────────────────
class Compose:
    def __init__(self, transforms_):
        self.transforms = transforms_

    def __call__(self, img, bboxes=None):
        for t in self.transforms:
            img, bboxes = t(img), bboxes
        return img, bboxes


# ───────────────────────────────────────────────────────────
# 3) 보조 함수
# ───────────────────────────────────────────────────────────
def draw_boxes_pil(image, boxes, conf_threshold):
    draw  = ImageDraw.Draw(image)
    font  = ImageFont.load_default()
    W, H  = image.size

    for cls, x, y, bw, bh, conf in boxes:
        if conf < conf_threshold or bw <= 0 or bh <= 0:
            continue

        # ➊ 중심좌표→코너좌표
        x1 = (x - bw / 2) * W
        y1 = (y - bh / 2) * H
        x2 = (x + bw / 2) * W
        y2 = (y + bh / 2) * H

        # ➋ 좌표 정렬
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))

        # ➌ 이미지 경계 클램프
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W - 1, x2), min(H - 1, y2)

        if x2 <= x1 or y2 <= y1:       # 여전히 말이 안 되면 스킵
            continue

        color = COLOR_MAP[int(cls)]
        label = f"{CLASS_NAMES[int(cls)]}: {conf:.2f}"

        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        if hasattr(draw, "textbbox"):
            tb = draw.textbbox((x1, y1), label, font=font)
            text_w, text_h = tb[2] - tb[0], tb[3] - tb[1]
        else:
            text_w, text_h = draw.textsize(label, font=font)

        draw.rectangle([x1, y1 - text_h, x1 + text_w, y1], fill=color)
        draw.text((x1, y1 - text_h), label, fill="black", font=font)

    return image


@torch.no_grad()
def predict_image(
    image_path: Path,
    model: torch.nn.Module,
    transform: Compose,
    args,
) -> Image.Image:
    orig_img = Image.open(image_path).convert("RGB")
    img, _ = transform(orig_img)
    img = img.unsqueeze(0).to(args.device)

    model.eval()
    preds = model(img)
    bboxes = cellboxes_to_boxes(preds, S=7)
    bboxes = non_max_suppression(
        bboxes[0],
        iou_threshold=args.iou_threshold,
        threshold=args.conf_threshold,
        box_format="midpoint",
    )

    return draw_boxes_pil(orig_img, bboxes, conf_threshold=args.conf_threshold)


# ───────────────────────────────────────────────────────────
# 4) main
# ───────────────────────────────────────────────────────────
def main(args) -> None:
    # ① 모델
    model = Yolov1(split_size=7, num_boxes=2, num_classes=20).to(args.device)
    ckpt = Path(args.weights_path)
    if not ckpt.exists():
        raise FileNotFoundError(ckpt)
    state = torch.load(ckpt, map_location=args.device)
    model.load_state_dict(state["state_dict"])
    print(f"✔️  Loaded weights: {ckpt}")
    print(f"‣ Device         : {args.device}")

    # ② 전처리
    transform = Compose(
        [
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
        ]
    )

    # ③ 입·출력 폴더
    inp_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    if not inp_dir.exists():
        raise FileNotFoundError(inp_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = sorted([p for p in inp_dir.iterdir()
                     if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    if not images:
        raise ValueError("입력 폴더에 이미지가 없습니다.")

    # ④ 추론 루프
    for img_path in images:
        print(f"[Inference] {img_path.name}")
        result_img = predict_image(img_path, model, transform, args)

        save_path = out_dir / f"{img_path.stem}_inference.jpg"
        result_img.save(save_path)
        print(f"   → Saved: {save_path}")


# ───────────────────────────────────────────────────────────
# 5) argparse 
# ───────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser("YOLOv1 Pascal-VOC 배치 추론")
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="'cpu' 또는 'cuda'",
    )
    p.add_argument(
        "--img_size",
        type=int,
        default=448,
        help="입력 리사이즈 크기",
    )
    p.add_argument(
        "--weights_path",
        default="checkpoints/best.pth",
        help="모델 가중치 파일(.pth or .pt) 경로",
    )
    p.add_argument(
        "--input_dir",
        default="data/example_images",
        help="입력 이미지 폴더",
    )
    p.add_argument(
        "--output_dir",
        default="data/inference_output",
        help="결과 저장 폴더",
    )
    p.add_argument(
        "--iou_threshold",
        type=float,
        default=0.3,
        help="IOU 임계값",
    )
    p.add_argument(
        "--conf_threshold",
        type=float,
        default=0.4,
        help="Confidence 임계값",
    )
    return p.parse_args()


# ───────────────────────────────────────────────────────────
# 6) 실행 엔트리
# ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()
    main(args)
