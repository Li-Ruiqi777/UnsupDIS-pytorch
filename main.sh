##### 从零在WarpedCOCO数据集对align进行预训练
python train.py --data data/warpedcoco.yaml --hyp data/hyp.align.scratch.yaml --cfg models/align_yolo.yaml --weights weights/yolov5x.pt --batch-size 16 --img-size 128 --epochs 150 --adam --device 0 --mode align
## variant network with HalfInstanceNormalizarion
# python train.py --data data/warpedcoco.yaml --hyp data/hyp.align.scratch.yaml --cfg models/align_variant.yaml --weights '' --batch-size 16 --img-size 128 --epochs 150 --adam --device 0 --mode align
## original network, not recommended, it is too slow
# python train.py --data data/warpedcoco.yaml --hyp data/hyp.align.scratch.yaml --cfg models/align_origin.yaml --weights '' --batch-size 4 --img-size 128 --epochs 150 --adam --device 0 --mode align

##### 在UDIS-D数据集对align微调
## `data/hyp.align.finetune.udis.yaml` is a very triky hyp-parameter set specified for UDIS-D. switch to `data/hyp.align.finetune.yaml` if necessary.
# python train.py --data data/udis.yaml --hyp data/hyp.align.finetune.udis.yaml --cfg models/align_yolo.yaml --weights weights/align/coco/weights/best.pt --batch-size 16 --img-size 128 --epochs 50 --adam --device 0 --mode align
#python train.py --data data/udis.yaml --hyp data/hyp.align.finetune.udis.yaml --cfg models/align_variant.yaml --weights weights/align/coco/weights/best.pt --batch-size 16 --img-size 128 --epochs 50 --adam --device 0 --mode align
python train.py --data data/udis.yaml --hyp data/hyp.align.finetune.udis.yaml --cfg models/align_origin.yaml --weights weights/align/coco/weights/best.pt --batch-size 4 --img-size 128 --epochs 50 --adam --device 0 --mode align

##### 生成粗配准的结果
python inference_align.py --weights weights/align/udis/weights/best.pt --source data/udis.yaml --task train
python inference_align.py --weights weights/align/udis/weights/best.pt --source data/udis.yaml --task test
mkdir UDIS-D/warp
mv runs/infer/exp UDIS-D/warp/train
mv runs/infer/exp2 UDIS-D/warp/test

##### 在UDIS-D数据集上训练Fusion模型
python train.py --data data/udis.yaml --hyp data/hyp.fuse.scratch.yaml --cfg models/fuse_yolo.yaml --weights weights/yolov5m.pt --batch-size 4 --img-size 640 --epochs 30 --adam --device 0 --mode fuse --reg-mode crop
## optional
# python train.py --data data/udis.yaml --hyp data/hyp.fuse.scratch.yaml --cfg models/fuse_yolo.yaml --weights weights/yolov5m.pt --batch-size 4 --img-size 512 --epochs 30 --adam --device 0 --mode fuse --reg-mode resize
# python train.py --data data/udis.yaml --hyp data/hyp.fuse.scratch.yaml --cfg models/fuse_origin.yaml --weights '' --batch-size 4 --img-size 512 --epochs 30 --adam --device 0 --mode fuse --reg-mode resize

##### 使用Fusion模型生成最后的拼接结果
python inference_fuse.py --weights weights/fuse/udis/weights/best.pt --source data/udis.yaml --task test --half --img-size 640 --reg-mode crop
# python inference_fuse.py --weights weights/fuse/udis/weights/best.pt --source data/udis.yaml --task test --half --img-size 512 --reg-mode resize