# 1.从零在UDIS-Ship数据集对align进行预训练
python train.py --data data/udis-ship.yaml --hyp data/hyp.align.finetune.udis.yaml --cfg models/align_origin.yaml --weights '' --batch-size 4 --img-size 512 --epochs 150 --adam --device 0 --mode align

# 2.生成粗配准的结果
python inference_align.py --weights weights/align/udis/weights/best.pt --source data/udis.yaml --task train
python inference_align.py --weights weights/align/udis/weights/best.pt --source data/udis.yaml --task test
mkdir UDIS-D/warp
mv runs/infer/exp UDIS-D/warp/train
mv runs/infer/exp2 UDIS-D/warp/test

# 3.在UDIS-D数据集上训练Fusion模型
python train.py --data data/udis.yaml --hyp data/hyp.fuse.scratch.yaml --cfg models/fuse_yolo.yaml --weights weights/yolov5m.pt --batch-size 4 --img-size 640 --epochs 30 --adam --device 0 --mode fuse --reg-mode crop
## optional
# python train.py --data data/udis.yaml --hyp data/hyp.fuse.scratch.yaml --cfg models/fuse_yolo.yaml --weights weights/yolov5m.pt --batch-size 4 --img-size 512 --epochs 30 --adam --device 0 --mode fuse --reg-mode resize
# python train.py --data data/udis.yaml --hyp data/hyp.fuse.scratch.yaml --cfg models/fuse_origin.yaml --weights '' --batch-size 4 --img-size 512 --epochs 30 --adam --device 0 --mode fuse --reg-mode resize

# 4.使用Fusion模型生成最后的拼接结果
python inference_fuse.py --weights weights/fuse/udis/weights/best.pt --source data/udis.yaml --task test --half --img-size 640 --reg-mode crop
# python inference_fuse.py --weights weights/fuse/udis/weights/best.pt --source data/udis.yaml --task test --half --img-size 512 --reg-mode resize