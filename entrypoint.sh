#!/bin/sh
#params="$@"
#echo parameters are $params
#process.py /input /output \


python3 process.py \
--input-dir /input --output-dir /output \
--config-file-maskr ./configs/COCO-InstanceSegmentation/mask_rcnn_pm_R_50_FPN_1x.yaml \
--config-file-retina ./configs/COCO-Detection/retinanet_pm_R_50_FPN_1x.yaml \
--num-gpus 1 