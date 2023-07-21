# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import torch
from torch_utils import persistence
from training.networks_stylegan2 import Generator as StyleGAN2Backbone
from training.volumetric_rendering.renderer import ImportanceRenderer
from training.volumetric_rendering.ray_sampler import RaySampler
import dnnlib
from torch_utils.ops import upfirdn2d

@persistence.persistent_class
class TriPlaneGenerator(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality.
        c_dim,                      # Conditioning label (C) dimensionality.
        w_dim,                      # Intermediate latent (W) dimensionality.
        triplane_resolution,
        triplane_channels,
        triplane_act,
        blur_plane,
        img_resolution,             # Output resolution.
        img_channels,               # Number of output color channels.
        decoder_output_dim  = 32,
        sr_num_fp16_res     = 0,
        mapping_kwargs      = {},   # Arguments for MappingNetwork.
        rendering_kwargs    = {},
        neural_rendering_resolution = 64,
        depth_guided_sample         = 0,
        roll_out            = None,
        sr_kwargs = {},
        **synthesis_kwargs,         # Arguments for SynthesisNetwork.
    ):
        super().__init__()
        self.z_dim=z_dim
        self.c_dim=c_dim
        self.w_dim=w_dim
        self.img_resolution=img_resolution
        self.img_channels=img_channels
        self.triplane_resolution=triplane_resolution
        self.triplane_channels=triplane_channels
        self.roll_out = roll_out if len(synthesis_kwargs['aware3d_res'])==0 else 'b'
        self.renderer = ImportanceRenderer()
        self.ray_sampler = RaySampler()
        self.backbone = StyleGAN2Backbone(z_dim, c_dim, w_dim, img_resolution=triplane_resolution, img_channels=triplane_channels//(1, 3)[roll_out in ['b', 'a']], mapping_kwargs=mapping_kwargs, roll_out=roll_out, **synthesis_kwargs)
        if rendering_kwargs['superresolution_module'] is not None and decoder_output_dim>3:
            self.superresolution = dnnlib.util.construct_class_by_name(class_name=rendering_kwargs['superresolution_module'], channels=32, img_resolution=img_resolution, sr_num_fp16_res=sr_num_fp16_res, sr_antialias=rendering_kwargs['sr_antialias'], **sr_kwargs)
        else:
             self.superresolution = None
        self.decoder = OSGDecoder(32, {'decoder_lr_mul': rendering_kwargs.get('decoder_lr_mul', 1), 'decoder_output_dim': decoder_output_dim})
        self.neural_rendering_resolution = neural_rendering_resolution
        self.rendering_kwargs = rendering_kwargs
        self.plane_act = None
        if triplane_act is not None:
            self.plane_act = torch.nn.Tanh() if triplane_act == 'tanh' else torch.nn.Sigmoid()
        self.blur_plane = blur_plane
        if blur_plane > 0:
            self.f = torch.arange(-blur_plane, blur_plane + 1).div(blur_plane).square().neg().exp2()
        else:
            self.f = None
        self._last_planes = None
    
    def mapping(self, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False):
        if self.rendering_kwargs['c_gen_conditioning_zero']:
                c = torch.zeros_like(c)
        return self.backbone.mapping(z, c * self.rendering_kwargs.get('c_scale', 0), truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)

    def synthesis(self, ws, c, neural_rendering_resolution=None, update_emas=False, cache_backbone=False, use_cached_backbone=False, chunk=None, ret_plane=False, patch_scale=1.0, run_full=True, blur_plane=0, **synthesis_kwargs):
        cam2world_matrix = c[:, :16].view(-1, 4, 4)
        intrinsics = c[:, 16:25].view(-1, 3, 3)
        N = cam2world_matrix.shape[0]

        if neural_rendering_resolution is None:
            neural_rendering_resolution = self.neural_rendering_resolution
        elif self.training:
            self.neural_rendering_resolution = neural_rendering_resolution
        H = W = neural_rendering_resolution

        # Create triplanes by running StyleGAN backbone
        if use_cached_backbone and self._last_planes is not None:
            planes = self._last_planes
        else:
            planes = self.backbone.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
            if self.plane_act is not None:
                planes = self.plane_act(planes)
        if cache_backbone:
            self._last_planes = planes

        # Reshape output into three 32-channel planes
        if self.f is not None:
            planes[2:] = upfirdn2d.filter2d(planes[2:], (self.f / self.f.sum()).to(planes.device))
        planes = self.plane_rehape(planes)
        
        output = {}
        if ret_plane:
            output.update({'plane': planes})
        
        if run_full:
            # Create a batch of rays for volume rendering
            ray_origins, ray_directions, _ = self.ray_sampler(cam2world_matrix, intrinsics, neural_rendering_resolution)
            
            # Perform volume rendering
            if chunk is None:
                feature_samples, depth_samples, weights_samples = self.renderer(planes, self.decoder, ray_origins, ray_directions, self.rendering_kwargs) # channels last
            else:
                feature_list, depth_list, weight_list = list(), list(), list() 
                for _ro, _rd in zip(torch.split(ray_origins, chunk, dim=1), torch.split(ray_directions, chunk, dim=1)):
                    _f, _d, _w = self.renderer(planes, self.decoder, _ro, _rd, self.rendering_kwargs)
                    feature_list.append(_f)
                    depth_list.append(_d)
                    weight_list.append(_w)
                feature_samples = torch.cat(feature_list, 1)
                depth_samples = torch.cat(depth_list, 1)
                weights_samples = torch.cat(weight_list, 1)

            # Reshape into 'raw' neural-rendered image
            H = W = neural_rendering_resolution
            feature_image = feature_samples.permute(0, 2, 1).reshape(N, feature_samples.shape[-1], H, W).contiguous()
            depth_image = depth_samples.permute(0, 2, 1).reshape(N, 1, H, W)

            # Run superresolution to get final image
            rgb_image = feature_image[:, :3]
            if self.superresolution is not None and rgb_image.shape[-1]<=self.superresolution.input_resolution:
                sr_image = self.superresolution(rgb_image, feature_image, ws, noise_mode=self.rendering_kwargs['superresolution_noise_mode'], **{k:synthesis_kwargs[k] for k in synthesis_kwargs.keys() if k != 'noise_mode'})
            else:
                sr_image = rgb_image
            
            output.update({'image': sr_image, 'image_raw': rgb_image, 'image_depth': depth_image})

        # patch
        if patch_scale<1:
            patch_ray_origins, patch_ray_directions, patch_info = self.ray_sampler(cam2world_matrix, intrinsics, neural_rendering_resolution, patch_scale=patch_scale)

            # Perform volume rendering
            if chunk is None:
                patch_feature_samples, patch_depth_samples, patch_weights_samples = self.renderer(planes, self.decoder, patch_ray_origins, patch_ray_directions, self.rendering_kwargs) # channels last
            else:
                patch_feature_list, patch_depth_list, patch_weight_list = list(), list(), list() 
                for _ro, _rd in zip(torch.split(patch_ray_origins, chunk, dim=1), torch.split(patch_ray_directions, chunk, dim=1)):
                    _f, _d, _w = self.renderer(planes, self.decoder, _ro, _rd, self.rendering_kwargs)
                    patch_feature_list.append(_f)
                    patch_depth_list.append(_d)
                    patch_weight_list.append(_w)
                patch_feature_samples = torch.cat(patch_feature_list, 1)
                patch_depth_samples = torch.cat(patch_depth_list, 1)
                patch_weights_samples = torch.cat(patch_weight_list, 1)

            # Reshape into 'raw' neural-rendered image
            patch_feature_image = patch_feature_samples.permute(0, 2, 1).reshape(N, patch_feature_samples.shape[-1], H, W).contiguous()
            patch_depth_image = patch_depth_samples.permute(0, 2, 1).reshape(N, 1, H, W)
            patch_rgb_image = patch_feature_image[:, :3]
            output.update({'patch_image_raw': patch_rgb_image})

            if run_full:
                patch_sr_image = []
                patch_rgb_image = []
                sr_image_ = sr_image.detach()
                rgb_image_ = rgb_image.detach()
                rgb_image_ = torch.nn.functional.interpolate(rgb_image_, size=(sr_image_.shape[-1]),
                                mode='bilinear', align_corners=False, antialias=True)
                for i in range(len(patch_info)):
                    top, left = patch_info[i]
                    patch_sr_image.append(sr_image_[i:i+1, :, top:top+neural_rendering_resolution, left:left+neural_rendering_resolution])
                    patch_rgb_image.append(rgb_image_[i:i+1, :, top:top+neural_rendering_resolution, left:left+neural_rendering_resolution])
                patch_sr_image = torch.cat(patch_sr_image, 0)
                patch_rgb_image = torch.cat(patch_rgb_image, 0)
                output.update({'patch_image': patch_sr_image, 'patch_image_gr': patch_rgb_image})

        return output
    
    def plane_rehape(self, planes):
        if self.roll_out=='w':
            planes = planes.view(len(planes), planes.shape[-3], planes.shape[-2], planes.shape[-1]//3, 3).permute(0, 4, 1, 2, 3).contiguous()
        elif self.roll_out in ['b', 'a']:
            planes = planes.view(len(planes)//3, 3, planes.shape[-3], planes.shape[-2], planes.shape[-1])
        else:
            planes = planes.view(len(planes), 3, planes.shape[-3]//3, planes.shape[-2], planes.shape[-1])
        
        return planes

    def sample(self, coordinates, directions, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
        # Compute RGB features, density for arbitrary 3D coordinates. Mostly used for extracting shapes. 
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        planes = self.backbone.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
        if self.plane_act is not None and not self.training:
            planes = self.plane_act(planes)
        planes = self.plane_rehape(planes)
        return self.renderer.run_model(planes, self.decoder, coordinates, directions, self.rendering_kwargs)

    def sample_mixed(self, coordinates, directions, ws, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
        # Same as sample, but expects latent vectors 'ws' instead of Gaussian noise 'z'
        planes = self.backbone.synthesis(ws, update_emas = update_emas, **synthesis_kwargs)
        if self.plane_act is not None and not self.training:
            planes = self.plane_act(planes)
        planes = self.plane_rehape(planes)
        return self.renderer.run_model(planes, self.decoder, coordinates, directions, self.rendering_kwargs)

    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, neural_rendering_resolution=None, update_emas=False, cache_backbone=False, use_cached_backbone=False, patch_scale=1.0,  chunk=None, **synthesis_kwargs):
        # Render a batch of generated images.
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        return self.synthesis(ws, c, update_emas=update_emas, neural_rendering_resolution=neural_rendering_resolution, cache_backbone=cache_backbone,use_cached_backbone=use_cached_backbone, patch_scale=patch_scale, chunk=chunk, **synthesis_kwargs)


from training.networks_stylegan2 import FullyConnectedLayer

class OSGDecoder(torch.nn.Module):
    def __init__(self, n_features, options):
        super().__init__()
        self.hidden_dim = 64

        self.net = torch.nn.Sequential(
            FullyConnectedLayer(n_features, self.hidden_dim, lr_multiplier=options['decoder_lr_mul']),
            torch.nn.Softplus(),
            FullyConnectedLayer(self.hidden_dim, 1 + options['decoder_output_dim'], lr_multiplier=options['decoder_lr_mul'])
        )
        
    def forward(self, sampled_features, ray_directions):
        # Aggregate features
        sampled_features = sampled_features.mean(1)
        x = sampled_features

        N, M, C = x.shape
        x = x.view(N*M, C)

        x = self.net(x)
        x = x.view(N, M, -1)
        rgb = torch.sigmoid(x[..., 1:])*(1 + 2*0.001) - 0.001 # Uses sigmoid clamping from MipNeRF
        sigma = x[..., 0:1]
        return {'rgb': rgb, 'sigma': sigma}
