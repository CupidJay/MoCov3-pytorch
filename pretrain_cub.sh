python main_moco_pretraining_v3.py \
  -a resnet50 \
  --lr 0.3 \
  --batch-size 256 --epochs 200 \
  --dist-url 'tcp://localhost:10005' --multiprocessing-distributed --world-size 1 --rank 0 \
  --gpus 8,9,10,11 \
  --mlp --moco-t 0.2 --moco-k 4096 --aug-plus --cos \
  /opt/caoyh/datasets/cub200
