# CUDA_VISIBLE_DEVICES=0,1,2,3 \
NVIDIA_TF32_OVERRIDE=0 \
python train.py \
    --cfg='configs/ffhq256_3d.yaml'
