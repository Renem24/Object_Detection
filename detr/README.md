# DETR : DEtection TRansformer
---

## Reference


---
## How to run MODEL?
We use `uv` and `argparse` to run python files with arguments. 

### Train
batch size = 8 기준, 20GB 정도 사용

```
uv run python3 detr/main.py \
    --batch_size 4 \
    --epochs 2 \
    --coco_path /mnt/ssd/lym/cvipl/dataset/COCO
    --output_dir detr/outputs
```

### Test

```
uv run python3 detr/main.py \
    --batch_size 2 \
    --no_aux_loss \
    --eval \
    --resume detr/checkpoints/detr-r50-e632da11.pth \
    --coco_path /mnt/ssd/lym/cvipl/dataset/COCO
```

### Inference

```

```

---


## Test results (COCO val 2017)
### Error Statistics
```
Averaged stats: 
    class_error: 14.29  
    loss: 1.0253 (1.1978)  
    loss_ce: 0.3285 (0.4904)  
    loss_bbox: 0.2005 (0.2246)  
    loss_giou: 0.4441 (0.4828)  
    loss_ce_unscaled: 0.3285 (0.4904)  
    class_error_unscaled: 17.8571 (22.0519)  
    loss_bbox_unscaled: 0.0401 (0.0449)  
    loss_giou_unscaled: 0.2221 (0.2414)  
    cardinality_error_unscaled: 4.0000 (5.0298)
```

### Average Precision & Recall
```
IoU metric: bbox
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.420
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.624
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.442
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.205
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.458
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.610
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.333
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.533
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.574
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.312
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.628
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.805
 ```