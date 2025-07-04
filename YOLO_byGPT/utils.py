"""Utility helpers: NMS, mAP, plotting, checkpoint save/load."""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import torch
import torchvision

from typing import List


def non_max_suppression(bboxes: List[List[float]], iou_threshold: float, threshold: float, box_format="corners"):
    bboxes = [box for box in bboxes if box[1] > threshold]
    bboxes.sort(key=lambda x: x[1], reverse=True)
    nms_boxes = []

    while bboxes:
        chosen_box = bboxes.pop(0)
        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0]
            or intersection_over_union(
                torch.tensor(chosen_box[2:]), torch.tensor(box[2:]), box_format=box_format
            ) < iou_threshold
        ]
        nms_boxes.append(chosen_box)
    return nms_boxes


def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    from loss import intersection_over_union as iou
    return iou(boxes_preds, boxes_labels, box_format)


def cellboxes_to_boxes(out, S=7):
    out = out.reshape(-1, S, S, 30)
    bboxes = []
    for n in range(out.shape[0]):
        bboxes_image = []
        for i in range(S):
            for j in range(S):
                if out[n, i, j, 20] > 0.5:
                    x, y, w, h = out[n, i, j, 21:25]
                    x = (x + j) / S
                    y = (y + i) / S
                    w /= S
                    h /= S
                    cls = torch.argmax(out[n, i, j, :20])
                    conf = out[n, i, j, 20]
                    bboxes_image.append([cls.item(), conf.item(), x.item(), y.item(), w.item(), h.item()])
        bboxes.append(bboxes_image)
    return bboxes


def plot_image(image, boxes):
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in range(20)]
    im = torchvision.transforms.ToPILImage()(image).convert("RGB")
    plt.figure()
    fig, ax = plt.subplots(1)
    ax.imshow(im)

    for box in boxes:
        cls, _, x, y, w, h = box
        x1, y1 = x - w / 2, y - h / 2
        rect = patches.Rectangle(
            (x1 * im.width, y1 * im.height), w * im.width, h * im.height, linewidth=2, edgecolor=colors[int(cls)], facecolor="none",
        )
        ax.add_patch(rect)
        ax.text(x1 * im.width, y1 * im.height, str(int(cls)), color="white", fontsize=8, backgroundcolor="black")
    plt.show()


def save_checkpoint(state, filename="checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint_file, model, optimizer=None, lr=None):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
        if lr is not None:
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
