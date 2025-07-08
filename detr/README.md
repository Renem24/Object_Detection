# DETR : DEtection TRansformer
---

## Reference

---
## Colab Notebooks

* [DETR's hands on Colab Notebook](https://colab.research.google.com/github/facebookresearch/detr/blob/colab/notebooks/detr_attention.ipynb): Shows how to load a model from hub, generate predictions, then visualize the attention of the model (similar to the figures of the paper)


* [Standalone Colab Notebook](https://colab.research.google.com/github/facebookresearch/detr/blob/colab/notebooks/detr_demo.ipynb): In this notebook, we demonstrate how to implement *a simplified version of DETR* from the grounds up in 50 lines of Python, then visualize the predictions. It is a good starting point if you want to gain better understanding the architecture and poke around before diving in the codebase.

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

# Model Zoo
We provide baseline DETR and DETR-DC5 models, and plan to include more in future.
AP is computed on COCO 2017 val5k, and inference time is over the first 100 val5k COCO images,
with torchscript transformer.

<table>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>backbone</th>
      <th>schedule</th>
      <th>inf_time</th>
      <th>box AP</th>
      <th>url</th>
      <th>size</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>DETR</td>
      <td>R50</td>
      <td>500</td>
      <td>0.036</td>
      <td>42.0</td>
      <td><a href="https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detr/logs/detr-r50_log.txt">logs</a></td>
      <td>159Mb</td>
    </tr>
    <tr>
      <th>1</th>
      <td>DETR-DC5</td>
      <td>R50</td>
      <td>500</td>
      <td>0.083</td>
      <td>43.3</td>
      <td><a href="https://dl.fbaipublicfiles.com/detr/detr-r50-dc5-f0fb7ef5.pth">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detr/logs/detr-r50-dc5_log.txt">logs</a></td>
      <td>159Mb</td>
    </tr>
    <tr>
      <th>2</th>
      <td>DETR</td>
      <td>R101</td>
      <td>500</td>
      <td>0.050</td>
      <td>43.5</td>
      <td><a href="https://dl.fbaipublicfiles.com/detr/detr-r101-2c7b67e5.pth">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detr/logs/detr-r101_log.txt">logs</a></td>
      <td>232Mb</td>
    </tr>
    <tr>
      <th>3</th>
      <td>DETR-DC5</td>
      <td>R101</td>
      <td>500</td>
      <td>0.097</td>
      <td>44.9</td>
      <td><a href="https://dl.fbaipublicfiles.com/detr/detr-r101-dc5-a2e86def.pth">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detr/logs/detr-r101-dc5_log.txt">logs</a></td>
      <td>232Mb</td>
    </tr>
  </tbody>
</table>

COCO val5k evaluation results can be found in this [gist](https://gist.github.com/szagoruyko/9c9ebb8455610958f7deaa27845d7918).

The models are also available via torch hub,
to load DETR R50 with pretrained weights simply do:
```python
model = torch.hub.load('facebookresearch/detr:main', 'detr_resnet50', pretrained=True)
```

---
## Data preparation

Download and extract COCO 2017 train and val images with annotations from
[http://cocodataset.org](http://cocodataset.org/#download).

This site's links don't work, so plz use these commands to download files!

### images
```
# images 
wget http://images.cocodataset.org/zips/train2017.zip # train dataset 
wget http://images.cocodataset.org/zips/val2017.zip # validation dataset 
wget http://images.cocodataset.org/zips/test2017.zip # test dataset 
wget http://images.cocodataset.org/zips/unlabeled2017.zip 
```

### labels(annotations)
```
# annotations 
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip 
wget http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip 
wget http://images.cocodataset.org/annotations/image_info_test2017.zip 
wget http://images.cocodataset.org/annotations/image_info_unlabeled2017.zip
```

We expect the directory structure to be the following:

```
path/to/coco/
  annotations/  # annotation json files
  train2017/    # train images
  val2017/      # val images
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