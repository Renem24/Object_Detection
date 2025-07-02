"""
Implementation of Yolo Loss Function from the original yolo paper
B=2, C=20를 가정하고 작성됨.
(grid cell당, box prediction 2개)
(output class 20개)
"""

import torch
import torch.nn as nn
from utils import intersection_over_union

class YoloLoss(nn.Module):

    def __init__(self, S=7, B=2, C=20):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")
        
        self.S = S
        self.B = B
        self.C = C
        
        self.lambda_noobj = 0.5
        self.lambda_coord = 5
        
    
    def forward(self, predictions, target):
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5)
        ## 5 - (x,y,w,h,confidence)
        
        ## default B=2이므로, iou_b1과 iou_b2를 비교하여, 높은 box를 "responsible"로 규정
        ## predictions  : predicted bbox
        ## target       : ground truth bbox
        iou_b1 = intersection_over_union(predictions[..., 21:25], target[..., 21:25])
        iou_b2 = intersection_over_union(predictions[..., 26:30], target[..., 21:25])
        
        ## .unsqueeze() : 지정한 dimension 위치에 길이가 1인 새로운 dimension 추가 
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)
        
        ## iou_maxes    : max values
        ## bestbox      : max indices
        ## torch.max(dim=0) : axis 0끼리(단위로) 비교
        iou_maxes, bestbox = torch.max(ious, dim=0)
        
        ## exists_box : 해당 grid cell 내, object의 존재 여부
        ##              object가 있는 cell만, loss를 계산하기 위해서 사용()
        ##              $\mathbb{1}_{i}^{\text{obj}}$
        ## target[..., 20]의 shape : (batch, S, S)
        ##      - 인덱스 0 ~ 19 : 클래스 원-핫/확률 20개 (0/1)
        ##      - 인덱스 20     : 객체 존재 여부 (0/1)
        exists_box = target[..., 20].unsqueeze(3)
        
        ## ================================================
        ##   BOX CENTER COORDINATES(x, y) & SIZE(w, h) LOSS 
        ## ================================================
        
        ## predicted bbox 
        box_predictions = exists_box * (
            (
                ## b0 : predictions[..., 21:25]
                ## b1 : predictions[..., 26:30]
                bestbox * predictions[..., 26:30]
                + (1 - bestbox) * predictions[..., 21:25]
            )
        )
        
        ## ground truth bbox
        box_targets = exists_box * target[..., 21:25]
        
        
        
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
            torch.abs(box_predictions[..., 2:4] + 1e-6)
        )
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])
        
        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2),
        )
        
        ## ================================================
        ##   FOR OBJECT LOSS (confidence)
        ## ================================================
        
        pred_box = (
            bestbox * predictions[..., 25:26] + (1 - bestbox) * predictions[..., 20:21]
        )
        
        object_loss = self.mse(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * target[..., 20:21]),
        )
        
        ## ================================================
        ##   FOR NO OBJECT LOSS (confidence)
        ## ================================================
        
        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 20:21], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21,], start_dim=1),
        )
        
        no_object_loss += self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 25:26], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21,], start_dim=1),
        )
        
        ## ================================================
        ##   FOR CLASS LOSS
        ## ================================================
        
        class_loss = self.mse(
            torch.flatten(exists_box * predictions[..., :20], end_dim=-2,),
            torch.flatten(exists_box * target[..., :20], end_dim=-2,),
        )
        
        loss = (
            self.lambda_coord * box_loss    # box coord & box size loss
            + object_loss
            + self.lambda_noobj * no_object_loss
            + class_loss
        )
        
        return loss