CUDA_VISIBLE_DEVICES=0 python train.py \
    --cfg='configs/ffhq512_3d.yaml' \
    phase 'infer' gpus 1
