"""Loss function for YOLOv1 (sum-squared error as in the original paper).

λ_coord = 5, λ_noobj = 0.5 by default.
Compute loss for a single batch of predictions and targets coming
from the model and VOCDataset respectively. Predictions are of shape
(BATCH, S, S, B*5 + C).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    """IoU between two sets of boxes (batch, 4)."""
    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2
    else:  # corners
        box1_x1, box1_y1, box1_x2, box1_y2 = boxes_preds[..., 0:1], boxes_preds[..., 1:2], boxes_preds[..., 2:3], boxes_preds[..., 3:4]
        box2_x1, box2_y1, box2_x2, box2_y2 = boxes_labels[..., 0:1], boxes_labels[..., 1:2], boxes_labels[..., 2:3], boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)


class YoloLoss(nn.Module):
    def __init__(self, S=7, B=2, C=20, lambda_coord: float = 5.0, lambda_noobj: float = 0.5):
        super().__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.S, self.B, self.C = S, B, C
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def forward(self, predictions, target):
        # predictions => (N, S, S, 5*B + C)
        N = predictions.shape[0]
        predictions = predictions.view(N, self.S, self.S, self.C + self.B * 5)

        # 존재하는 객체에 대한 마스크
        obj_mask = target[..., 4].unsqueeze(-1)  # (N,S,S,1)
        noobj_mask = 1 - obj_mask

        # ---------------- objeto 없는 셀 ----------------
        noobj_loss = self.mse(
            predictions[..., self.C + 4::5] * noobj_mask, target[..., 4::5] * noobj_mask,
        )

        # ---------------- 객체 있는 셀 ----------------
        # 선택적: 두 박스 중 IoU가 큰 박스 하나만 책임짐
        ious = []
        for b in range(self.B):
            pred_box = predictions[..., self.C + 5 * b : self.C + 5 * b + 4]
            ious.append(intersection_over_union(pred_box, target[..., self.C + 5 * 0 : self.C + 5 * 0 + 4]))
        ious = torch.stack(ious, dim=0)  # (B, N,S,S,1)
        best_box = ious.argmax(dim=0).unsqueeze(-1)  # (N,S,S,1)

        coord_loss = 0
        conf_loss = 0
        for b in range(self.B):
            mask_b = (best_box == b).float()
            pred_box = predictions[..., self.C + 5 * b : self.C + 5 * b + 4]
            targ_box = target[..., self.C + 5 * 0 : self.C + 5 * 0 + 4]  # GT coords broadcasted

            # 좌표
            coord_loss += self.mse(
                mask_b * obj_mask * pred_box[..., 0:2], mask_b * obj_mask * targ_box[..., 0:2]
            )
            # 루트로 스케일 보정된 너비/높이
            coord_loss += self.mse(
                mask_b * obj_mask * torch.sign(pred_box[..., 2:4]) * torch.sqrt(pred_box[..., 2:4].clamp(0)),
                mask_b * obj_mask * torch.sqrt(targ_box[..., 2:4])
            )
            # confidence
            pred_conf = predictions[..., self.C + 5 * b + 4 : self.C + 5 * b + 5]
            targ_conf = obj_mask * ious[b].detach()
            conf_loss += self.mse(mask_b * pred_conf, mask_b * targ_conf)

        class_loss = self.mse(obj_mask * predictions[..., :self.C], obj_mask * target[..., :self.C])

        loss = (
            self.lambda_coord * coord_loss
            + conf_loss
            + self.lambda_noobj * noobj_loss
            + class_loss
        ) / N
        return loss