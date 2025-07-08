# YOLO
---

## Reference
- https://arxiv.org/pdf/1506.02640   ← YOLOv1 paper
- https://arxiv.org/pdf/2304.00501   ← YOLO survey paper
- https://www.youtube.com/watch?v=n9_XyCGr-MI
- https://github.com/sangnekim/YOLO-v1-for-studying
- https://github.com/tanjeffreyz/yolo-v1   ← code implementation
- https://github.com/sangnekim/YOLO-v1-for-studying   ← code implementation
- https://github.com/motokimura/yolo_v1_pytorch   ← code implementation

---
## How to run MODEL?
We use `argparse` to run python files with arguments. 

### Train

```
uv run python3 yolov1_torch/train.py \
	--use_csv  \
	--train_csv /mnt/ssd/lym/cvipl/dataset/PascalVOC_YOLO/100examples.csv \
	--val_csv /mnt/ssd/lym/cvipl/dataset/PascalVOC_YOLO/test.csv \
	--img_dir /mnt/ssd/lym/cvipl/dataset/PascalVOC_YOLO/images \
	--label_dir /mnt/ssd/lym/cvipl/dataset/PascalVOC_YOLO/labels \
	--epochs 1000
```

### Test

```
uv run python3 yolov1_torch/test.py \
	--test_csv /mnt/ssd/lym/cvipl/dataset/PascalVOC_YOLO/test.csv \
	--img_dir /mnt/ssd/lym/cvipl/dataset/PascalVOC_YOLO/images \
	--label_dir /mnt/ssd/lym/cvipl/dataset/PascalVOC_YOLO/labels 
```

### Inference

```
uv run python3 yolov1_torch/inference.py \
    --weights_path yolov1_torch/checkpoints/yolo_1.pth \
    --input_dir yolov1_torch/sample_images \
    --output_dir yolov1_torch/outputs
```

---



## LOSS Explained

### **"Responsible" bounding box** selection
YOLO는 각 grid cell당, 여러($B$ 개의) bounding boxes을 predict한다. 

(논문 구현으로는 각 grid cell당 2개의 box를 predict)

bounding box predictor는 **하나의 object에, 하나의 bounding box만 predict**해야 하기 때문에, 여러 개의 bounding box 중 하나를 선택

ground truth bounding box와 IOU가 가장 높은 bounding box prediction을 **"responsible" bounding box**로 선택


### Loss Function
$$
\begin{aligned}
\text{Loss} = & \lambda_{\text{coord}} \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{\text{obj}} \left[ (x_i - \hat{x}_i)^2 + (y_i - \hat{y}_i)^2 \right] \\
& + \lambda_{\text{coord}} \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{\text{obj}} \left[ (\sqrt{w_i} - \sqrt{\hat{w}_i})^2 + (\sqrt{h_i} - \sqrt{\hat{h}_i})^2 \right] \\
& + \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{\text{obj}} (C_i - \hat{C}_i)^2 \\
& + \lambda_{\text{noobj}} \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{\text{noobj}} (C_i - \hat{C}_i)^2 \\
& + \sum_{i=0}^{S^2} \mathbb{1}_i^{\text{obj}} \sum_{c \in \text{classes}} (p_i(c) - \hat{p}_i(c))^2s
\end{aligned}
\tag{3}
$$

### notation
- $\mathbb{1}_{i}^{\text{obj}}$ : grid cell $i$에 object가 있으면 1, 아니면 0
- $\mathbb{1}_{ij}^{\text{obj}}$ : grid cell $i$에, $j$ 번째 bounding box가 "responsible"하면 1, 아니면 0

### 식 설명
1. bounding box coord.(x, y) loss
- bounding box가 ground truth box에 대해 "responsible"한 경우에만, bounding box coord. loss를 적용
	- 그 grid cell에서 가장 높은 IOU를 가진 box를 비교

$$\lambda_{\text{coord}} \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{\text{obj}} \left[ (x_i - \hat{x}_i)^2 + (y_i - \hat{y}_i)^2 \right]$$

2. bounding box size(width & height) loss
$$\lambda_{\text{coord}} \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{\text{obj}} \left[ (\sqrt{w_i} - \sqrt{\hat{w}_i})^2 + (\sqrt{h_i} - \sqrt{\hat{h}_i})^2 \right]$$

3. bounding box confidence loss (object)
$$\sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{\text{obj}} (C_i - \hat{C}_i)^2$$

4. bounding box confidence loss (no object)
$$\lambda_{\text{noobj}} \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{\text{noobj}} (C_i - \hat{C}_i)^2$$

5. classification loss
- object가 grid cell에 있는 경우에만, classification error를 적용
$$\sum_{i=0}^{S^2} \mathbb{1}_i^{\text{obj}} \sum_{c \in \text{classes}} (p_i(c) - \hat{p}_i(c))^2$$



## YOLO의 label(ground truth) format
각 셀당
- 인덱스 0 ~ 19     : 클래스 원-핫/확률 20개 (0 or 1)
- 인덱스 20         : 객체 존재 여부(confidence) (0 or 1)
- 인덱스 21 ~ 24    : GT 박스(x, y, w, h)

