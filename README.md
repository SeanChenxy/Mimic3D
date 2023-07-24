# Mimic3D: Thriving 3D-Aware GANs via 3D-to-2D Imitation (ICCV 2023)

[Project Page](https://seanchenxy.github.io/Mimic3DWeb/) | [Paper](https://arxiv.org/abs/2303.09036) 

# Requirements and Data Preparation
+ Our code is adopted from [EG3D](https://github.com/NVlabs/eg3d) and follow its requirements and data preparation.
+ Create a environment
    ```bash
    conda env create -f environment.yml
    conda activate eg3d
    ```
+ Follow [EG3D](https://github.com/NVlabs/eg3d/tree/main/dataset_preprocessing) to pre-process FFHQ, AFHQ, and ShapeNet data.
+ Pretrained models are avaliable at [Google Drive](https://drive.google.com/drive/folders/1zu9PUD2TvPuc-zTU1hK8q8GnnxcUdZxj?usp=sharing).
+ The data and model floders look as follows:
    ```
    ROOT
        ├──data
            ├──AFHQ
                ├── adhq_v2_256.zip
                ├── adhq_v2_512.zip
            ├──FFHQ
                ├──FFHQ_256.zip
                ├──FFHQ_512.zip
            ├──ShapeNet
                |──car_128.zip
        ├──out
            ├──afhq256_2d
            ├──afhq256_3d
            ├──afhq512_2d
            ├──afhq512_3d
            ├──ffhq256_2d
            ├──ffhq256_3d
            ├──ffhq512_2d
            ├──ffhq512_3d
            ├──shapenet128_2d
            ├──shapenet128_3d
    ```
# Inference
```
./scripts/infer.sh
```
+ Results will save to `out/{experiment}/infer`


# Evaluation 
```
./scripts/val.sh
```

# Training 
```
./scripts/train.sh
```

# Config file
+ In above `sh` files, `--cfg` can be changed for different models.
+ In a config file (e.g., configs/ffhq_3d.yaml), key settings are explained as follows:
    ```yaml
    experiment: 'ffhq512_3d' # your experiment name
    aware3d_res: [4,8,16,32,64,128,256] # the resolution with 3D-aware conv
    resume: '017000' # model to load; None for from-scratch; we suggest loading a 2D model before training a 3D model; also, a 3D model can be trained from scratch
    patch_gan: 0.1 # loss weight for patch distriminator
    metrics: [] # For 512-size model w/o 2D super-res., FID evaluation takes ~6h. You would want to set `metrics: []` to cancel evaluation durning training
    inference_mode: 'video' # select from video|videos|image
    neural_rendering_resolution_infer: 512 # rendering resolution with radiance field
    coarse: 0 # which tri-plane used for rendering. 0: coarse & detail triplanes; 1: coarse triplane; 2: detail triplane
    retplane: -1 # return triplane or not
    shapes: False # extract geometry or not
    ```
