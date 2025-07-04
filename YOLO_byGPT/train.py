"""Training script for YOLOv1 on Pascal VOC.
Usage:
$ python train.py --csv data/train.csv --epochs 200
"""

import argparse
from pathlib import Path
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import VOCDataset
from model import Yolov1
from loss import YoloLoss
from utils import (
    non_max_suppression,
    cellboxes_to_boxes,
    save_checkpoint,
    load_checkpoint,
)



def train_fn(train_loader, model, optimizer, loss_fn, scaler, device):
    loop = tqdm(train_loader, leave=False)
    losses = []
    for x, y in loop:
        x, y = x.to(device), y.to(device)
        with torch.cuda.amp.autocast():
            out = model(x)
            loss = loss_fn(out, y)
        losses.append(loss.item())

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loop.set_postfix(loss=sum(losses) / len(losses))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("--img_dir", type=str, default=None)
    parser.add_argument("--label_dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--weights", type=str, default="")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataset = (
        VOCDataset(csv_file=args.csv)
        if args.csv
        else VOCDataset(img_dir=args.img_dir, label_dir=args.label_dir)
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True)

    model = Yolov1().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = YoloLoss()
    scaler = torch.cuda.amp.GradScaler()

    if args.weights:
        load_checkpoint(args.weights, model, optimizer, args.lr)

    for epoch in range(args.epochs):
        print(f"Epoch {epoch}/{args.epochs}")
        train_fn(train_loader, model, optimizer, loss_fn, scaler, device)

        # save periodically
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, filename=f"checkpoint_epoch{epoch+1}.pth.tar")

    # final save
    save_checkpoint({"state_dict": model.state_dict()}, filename="yolov1_final.pth.tar")
