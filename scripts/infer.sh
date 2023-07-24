CUDA_VISIBLE_DEVICES=0 /cpfs/user/chenxingyu/.conda/envs/pt111_mnt/bin/python train.py \
    --cfg='configs/afhq512_3d.yaml' \
    phase 'infer' gpus 1
