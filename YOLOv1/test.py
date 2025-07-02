import torch
from torch.utils.data import DataLoader
from model import Yolov1
from dataset import VOCDataset
from utils import mean_average_precision, get_bboxes, non_max_suppression

# 하이퍼파라미터와 경로 설정
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
IMG_DIR = "data/images"
LABEL_DIR = "data/labels"
WEIGHT_PATH = "best.pth.tar"  # 사용할 모델 가중치 파일

# 데이터셋 및 로더
test_dataset = VOCDataset(
    "data/test.csv", 
    transform=None,       # 필요한 경우 transform 추가
    img_dir=IMG_DIR,
    label_dir=LABEL_DIR,
)
test_loader = DataLoader(
    dataset=test_dataset, 
    batch_size=BATCH_SIZE,
    shuffle=False
)

# 모델 불러오기
model = Yolov1(split_size=7, num_boxes=2, num_classes=20).to(DEVICE)
model.load_state_dict(torch.load(WEIGHT_PATH, map_location=DEVICE))
model.eval()

all_pred_boxes = []
all_true_boxes = []

# 테스트 루프
for batch_idx, (x, y) in enumerate(test_loader):
    x = x.to(DEVICE)
    with torch.no_grad():
        preds = model(x)
    batch_boxes = get_bboxes(preds, ...)
    all_pred_boxes += batch_boxes
    all_true_boxes += ...  # 정답 박스 추가

# 평가 (mAP 등)
mAP = mean_average_precision(all_pred_boxes, all_true_boxes, ...)
print(f"mAP: {mAP}")