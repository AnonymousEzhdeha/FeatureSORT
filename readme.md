

## Dataset 

- Annotation: prepare annotation in  yolo style i.e 1 .txt file for 1 image. Annotation format cls_id, xc, yc, w, h. Check samples in `data/sample_data/labels/`
- Specify train and validation set. 1 image - annotation file is save in 1 line in txt file. Check “data/sample_data/train.txt” for details.
- Then create a config file. The config file contains information about train/val set, number of classes, names. Check data/map_test.yaml for detail.

##Train

Run `CUDA_VISIBLE_DEVICES=0 python train.py --img-size 1344 --epochs 30 --batch-size 3 --data data/map_test.yaml --cfg models/phrd_cosinenet.yaml --weights phrd_coco.pt --project runs/sample_train`
 - outputs (logs, model weight, etc) will be saved in `runs/sample_train`
