# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Loss functions."""

import numpy as np
import torch
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import upfirdn2d
from training.dual_discriminator import filtered_resizing

#----------------------------------------------------------------------------

class Loss:
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg): # to be overridden by subclass
        raise NotImplementedError()

#----------------------------------------------------------------------------

class StyleGAN2Loss(Loss):
    def __init__(
        self, device, G, D, 
        D_patch=None, 
        augment_pipe=None, 
        lpips=None, 
        r1_gamma=10, 
        r1_gamma_patch=10,
        style_mixing_prob=0, 
        pl_weight=0, 
        pl_batch_shrink=2, 
        pl_decay=0.01, 
        pl_no_weight_grad=False, 
        blur_init_sigma=0, 
        blur_init_sigma_patch=0, 
        blur_fade_kimg=0, 
        r1_gamma_init=0, 
        r1_gamma_fade_kimg=0, 
        neural_rendering_resolution_initial=64, 
        neural_rendering_resolution_final=None, 
        neural_rendering_resolution_fade_kimg=0, 
        gpc_reg_fade_kimg=1000, 
        gpc_reg_prob=None, 
        dual_discrimination=False, 
        dual_patch_discrimination = False,
        filter_mode='antialiased', 
        patch_scale=1.0, 
        feat_match=0.0,
        detail_reg=0.0,
        patch_gan=0.0,
    ):
        super().__init__()
        self.device             = device
        self.G                  = G
        self.D                  = D
        self.D_patch            = D_patch
        self.augment_pipe       = augment_pipe
        self.lpips              = lpips
        self.r1_gamma           = r1_gamma
        self.r1_gamma_patch     = r1_gamma_patch
        self.style_mixing_prob  = style_mixing_prob
        self.pl_weight          = pl_weight
        self.pl_batch_shrink    = pl_batch_shrink
        self.pl_decay           = pl_decay
        self.pl_no_weight_grad  = pl_no_weight_grad
        self.pl_mean            = torch.zeros([], device=device)
        self.blur_init_sigma    = blur_init_sigma
        self.blur_init_sigma_patch = blur_init_sigma_patch
        self.blur_fade_kimg     = blur_fade_kimg
        self.r1_gamma_init      = r1_gamma_init
        self.r1_gamma_fade_kimg = r1_gamma_fade_kimg
        self.neural_rendering_resolution_initial = neural_rendering_resolution_initial
        self.neural_rendering_resolution_final = neural_rendering_resolution_final
        self.neural_rendering_resolution_fade_kimg = neural_rendering_resolution_fade_kimg
        self.gpc_reg_fade_kimg = gpc_reg_fade_kimg
        self.gpc_reg_prob = gpc_reg_prob
        self.dual_discrimination = dual_discrimination
        self.dual_patch_discrimination = dual_patch_discrimination
        self.filter_mode = filter_mode
        self.resample_filter = upfirdn2d.setup_filter([1,3,3,1], device=device)
        self.blur_raw_target = True
        assert self.gpc_reg_prob is None or (0 <= self.gpc_reg_prob <= 1)
        self.patch_scale = patch_scale
        self.feat_match = feat_match
        self.detail_reg = detail_reg
        self.patch_gan = patch_gan

    def run_G(self, z, c, swapping_prob, neural_rendering_resolution, update_emas=False, patch_scale=1.0, run_full=True, ret_plane=False):
        if swapping_prob is not None:
            c_swapped = torch.roll(c.clone(), 1, 0)
            c_gen_conditioning = torch.where(torch.rand((c.shape[0], 1), device=c.device) < swapping_prob, c_swapped, c)
        else:
            c_gen_conditioning = torch.zeros_like(c)

        ws = self.G.mapping(z, c_gen_conditioning, update_emas=update_emas)
        if self.style_mixing_prob > 0:
            with torch.autograd.profiler.record_function('style_mixing'):
                cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                ws[:, cutoff:] = self.G.mapping(torch.randn_like(z), c, update_emas=False)[:, cutoff:]
        gen_output = self.G.synthesis(ws, c, neural_rendering_resolution=neural_rendering_resolution, update_emas=update_emas, patch_scale=patch_scale, run_full=run_full, ret_plane=ret_plane)
        return gen_output, ws

    def run_D(self, img, c, blur_sigma=0, blur_sigma_raw=0, update_emas=False):
        blur_size = np.floor(blur_sigma * 3)
        if blur_size > 0:
            with torch.autograd.profiler.record_function('blur'):
                f = torch.arange(-blur_size, blur_size + 1, device=img['image'].device).div(blur_sigma).square().neg().exp2()
                img['image'] = upfirdn2d.filter2d(img['image'], f / f.sum())

        if self.augment_pipe is not None:
            augmented_pair = self.augment_pipe(torch.cat([img['image'],
                                                    torch.nn.functional.interpolate(img['image_raw'], size=img['image'].shape[2:], mode='bilinear', antialias=True)],
                                                    dim=1))
            img['image'] = augmented_pair[:, :img['image'].shape[1]]
            img['image_raw'] = torch.nn.functional.interpolate(augmented_pair[:, img['image'].shape[1]:], size=img['image_raw'].shape[2:], mode='bilinear', antialias=True)

        logits = self.D(img, c, update_emas=update_emas)
        return logits
    
    def run_D_patch(self, img, blur_sigma=0, blur_sigma_raw=0, feat_match=0.0, run_patch=True, run_patch_raw=True, update_emas=False):
        blur_size = np.floor(blur_sigma * 3)
        diff_feat, patch_raw_logits, patch_logits = None, None, None
        if feat_match > 0:
            patch_logits, patch_feat = self.D_patch(img, None, img_name='patch_image', retfeat=True, update_emas=update_emas)
            patch_raw_logits, patch_raw_feat = self.D_patch(img, None, img_name='patch_image_raw', retfeat=True, update_emas=update_emas)
            patch_feat = [self.normalize_tensor(p) for p in patch_feat]
            patch_raw_feat = [self.normalize_tensor(p) for p in patch_raw_feat]
            diff_feat = [((p-pr)**2).mean([1,2,3]) for p, pr in zip(patch_feat, patch_raw_feat)]
            diff_feat = sum(diff_feat)[:, None] * feat_match
        else:
            if run_patch:
                if blur_size > 0:
                    with torch.autograd.profiler.record_function('blur_patch'):
                        f = torch.arange(-blur_size, blur_size + 1, device=img['patch_image'].device).div(blur_sigma).square().neg().exp2()
                        img['patch_image'] = upfirdn2d.filter2d(img['patch_image'], f / f.sum())
                if self.augment_pipe is not None:
                    img['patch_image'] = self.augment_pipe(img['patch_image'])
                patch_logits = self.D_patch(img, None, img_name='patch_image', img_raw_name='patch_image_raw', retfeat=False, update_emas=update_emas)
            if run_patch_raw:
                if blur_size > 0:
                    with torch.autograd.profiler.record_function('blur_patch'):
                        f = torch.arange(-blur_size, blur_size + 1, device=img['patch_image_raw'].device).div(blur_sigma).square().neg().exp2()
                        img['patch_image_raw'] = upfirdn2d.filter2d(img['patch_image_raw'], f / f.sum())
                if self.augment_pipe is not None:
                    img['patch_image_raw'] = self.augment_pipe(img['patch_image_raw'])
                patch_raw_logits = self.D_patch(img, None, img_name='patch_image_raw', img_raw_name='patch_image_gr', retfeat=False, update_emas=update_emas)
        return patch_logits, patch_raw_logits, diff_feat

    @staticmethod
    def normalize_tensor(in_feat, eps=1e-10):
        norm_factor = torch.sqrt(torch.sum(in_feat**2,dim=1,keepdim=True)+eps)
        return in_feat/(norm_factor+eps)

    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg):
        # assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        if self.G.rendering_kwargs.get('density_reg', 0) == 0:
            phase = {'Greg': 'none', 'Gboth': 'Gmain'}.get(phase, phase)
        if self.r1_gamma == 0:
            phase = {'Dreg': 'none', 'Dboth': 'Dmain'}.get(phase, phase)
        if self.r1_gamma_patch == 0:
            phase = {'D_patchreg': 'none', 'D_patchboth': 'Dmain'}.get(phase, phase)
        blur_sigma = max(1 - cur_nimg / (self.blur_fade_kimg * 1e3), 0) * self.blur_init_sigma if self.blur_fade_kimg > 0 else 0
        blur_sigma_patch = max(1 - cur_nimg / (self.blur_fade_kimg * 1e3), 0) * self.blur_init_sigma_patch if self.blur_fade_kimg > 0 else 0
        r1_gamma = self.r1_gamma
        r1_gamma_patch = self.r1_gamma_patch
        training_stats.report('Progress/blur_sigma', blur_sigma)
        training_stats.report('Progress/blur_sigma_patch', blur_sigma_patch)

        alpha = min(cur_nimg / (self.gpc_reg_fade_kimg * 1e3), 1) if self.gpc_reg_fade_kimg > 0 else 1
        swapping_prob = (1 - alpha) * 1 + alpha * self.gpc_reg_prob if self.gpc_reg_prob is not None else None

        if self.neural_rendering_resolution_final is not None:
            alpha = min(cur_nimg / (self.neural_rendering_resolution_fade_kimg * 1e3), 1)
            neural_rendering_resolution = int(np.rint(self.neural_rendering_resolution_initial * (1 - alpha) + self.neural_rendering_resolution_final * alpha))
        else:
            neural_rendering_resolution = self.neural_rendering_resolution_initial
        training_stats.report('Progress/render_size', neural_rendering_resolution)

        real_img_raw = filtered_resizing(real_img, size=neural_rendering_resolution, f=self.resample_filter, filter_mode=self.filter_mode)

        if self.blur_raw_target:
            blur_size = np.floor(blur_sigma * 3)
            if blur_size > 0:
                f = torch.arange(-blur_size, blur_size + 1, device=real_img_raw.device).div(blur_sigma).square().neg().exp2()
                real_img_raw = upfirdn2d.filter2d(real_img_raw, f / f.sum())

        real_img = {'image': real_img, 'image_raw': real_img_raw}
        if 'patch' in phase:
            real_img_tmp = real_img['image'].clone()
            real_img_raw_tmp = real_img['image_raw'].clone()
            blur_size_patch = np.floor(blur_sigma_patch * 3)
            if blur_size_patch > 0:
                f = torch.arange(-blur_size_patch, blur_size_patch + 1, device=real_img_raw_tmp.device).div(blur_sigma_patch).square().neg().exp2()
                real_img_raw_tmp = upfirdn2d.filter2d(real_img_raw_tmp, f / f.sum())
            real_img_raw_tmp = torch.nn.functional.interpolate(real_img_raw_tmp, size=(real_img_tmp.shape[-1]),
                                mode='bilinear', align_corners=False, antialias=True)
            real_img_patch = []
            real_img_patch_raw = []
            for i in range(real_img_tmp.shape[0]):
                top = torch.randint(real_img_tmp.shape[-1]-neural_rendering_resolution+1, ()).item()
                left = torch.randint(real_img_tmp.shape[-1]-neural_rendering_resolution+1, ()).item()
                real_img_patch.append(real_img_tmp[i:i+1, :, top:top+neural_rendering_resolution, left:left+neural_rendering_resolution])
                real_img_patch_raw.append(real_img_raw_tmp[i:i+1, :, top:top+neural_rendering_resolution, left:left+neural_rendering_resolution])
            real_img_patch = torch.cat(real_img_patch, 0)
            real_img_patch_raw = torch.cat(real_img_patch_raw, 0)
            real_img.update({'patch_image': real_img_patch, 'patch_image_raw': real_img_patch_raw})

        # Gmain: Maximize logits for generated images.
        if phase in ['Gmain', 'Gboth']:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                gen_img, _gen_ws = self.run_G(
                    gen_z, gen_c, swapping_prob=swapping_prob, 
                    neural_rendering_resolution=neural_rendering_resolution, 
                    patch_scale=self.patch_scale, 
                    run_full=self.dual_patch_discrimination or self.lpips is not None or self.D is not None,
                    ret_plane=self.detail_reg>0)
                loss_Gmain, loss_Glpips, loss_Gpatch, loss_Gfeat, loss_Gdetail = None, None, None, None, None
                if self.detail_reg > 0:
                    loss_Gdetail = torch.abs(gen_img['plane'][-1]).mean() * self.detail_reg
                    training_stats.report('Loss/G/detail_reg', loss_Gdetail)
                if self.D is not None:
                    gen_logits = self.run_D(gen_img, gen_c, blur_sigma=blur_sigma)
                    training_stats.report('Loss/scores/fake', gen_logits)
                    training_stats.report('Loss/signs/fake', gen_logits.sign())
                    loss_Gmain = torch.nn.functional.softplus(-gen_logits)
                    training_stats.report('Loss/G/loss', loss_Gmain)
                if 'patch_image_raw' in gen_img:
                    if self.lpips is not None:
                        loss_Glpips = self.lpips(gen_img['patch_image_raw'], gen_img['patch_image'])
                        training_stats.report('Loss/G/lpips', loss_Glpips)
                    if self.D_patch is not None and self.patch_gan>0:
                        _, patch_raw_logits, loss_Gfeat = self.run_D_patch(gen_img, blur_sigma=blur_sigma, run_patch=False, feat_match=self.feat_match)
                        training_stats.report('Loss/scores/patch_fake', patch_raw_logits)
                        training_stats.report('Loss/signs/patch_fake', patch_raw_logits.sign())
                        loss_Gpatch = torch.nn.functional.softplus(-patch_raw_logits) * self.patch_gan
                        training_stats.report('Loss/G/patch', loss_Gpatch)
                        if loss_Gfeat is not None:
                            training_stats.report('Loss/G/feat', loss_Gfeat)                                     
                                    
            with torch.autograd.profiler.record_function('Gmain_backward'):
                loss_G = 0.
                if loss_Gmain is not None:
                    loss_G += loss_Gmain.mean()
                if loss_Glpips is not None:
                    loss_G += loss_Glpips.mean()
                if loss_Gpatch is not None:
                    loss_G += loss_Gpatch.mean()
                if loss_Gfeat is not None:
                    loss_G += loss_Gfeat.mean()
                if loss_Gdetail is not None:
                    loss_G += loss_Gdetail.mean()
                loss_G.mul(gain).backward()

        # Density Regularization
        if phase in ['Greg', 'Gboth'] and self.G.rendering_kwargs.get('density_reg', 0) > 0 and self.G.rendering_kwargs['reg_type'] == 'l1':
            if swapping_prob is not None:
                c_swapped = torch.roll(gen_c.clone(), 1, 0)
                c_gen_conditioning = torch.where(torch.rand([], device=gen_c.device) < swapping_prob, c_swapped, gen_c)
            else:
                c_gen_conditioning = torch.zeros_like(gen_c)

            ws = self.G.mapping(gen_z, c_gen_conditioning, update_emas=False)
            if self.style_mixing_prob > 0:
                with torch.autograd.profiler.record_function('style_mixing'):
                    cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                    cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                    ws[:, cutoff:] = self.G.mapping(torch.randn_like(z), c, update_emas=False)[:, cutoff:]
            initial_coordinates = torch.rand((ws.shape[0], 1000, 3), device=ws.device) * 2 - 1
            perturbed_coordinates = initial_coordinates + torch.randn_like(initial_coordinates) * self.G.rendering_kwargs['density_reg_p_dist']
            all_coordinates = torch.cat([initial_coordinates, perturbed_coordinates], dim=1)
            sigma = self.G.sample_mixed(all_coordinates, torch.randn_like(all_coordinates), ws, update_emas=False)['sigma']
            sigma_initial = sigma[:, :sigma.shape[1]//2]
            sigma_perturbed = sigma[:, sigma.shape[1]//2:]

            TVloss = torch.nn.functional.l1_loss(sigma_initial, sigma_perturbed) * self.G.rendering_kwargs['density_reg']
            training_stats.report('Loss/G/reg', TVloss)
            TVloss.mul(gain).backward()

        # Alternative density regularization
        if phase in ['Greg', 'Gboth'] and self.G.rendering_kwargs.get('density_reg', 0) > 0 and self.G.rendering_kwargs['reg_type'] == 'monotonic-detach':
            if swapping_prob is not None:
                c_swapped = torch.roll(gen_c.clone(), 1, 0)
                c_gen_conditioning = torch.where(torch.rand([], device=gen_c.device) < swapping_prob, c_swapped, gen_c)
            else:
                c_gen_conditioning = torch.zeros_like(gen_c)

            ws = self.G.mapping(gen_z, c_gen_conditioning, update_emas=False)

            initial_coordinates = torch.rand((ws.shape[0], 2000, 3), device=ws.device) * 2 - 1 # Front

            perturbed_coordinates = initial_coordinates + torch.tensor([0, 0, -1], device=ws.device) * (1/256) * self.G.rendering_kwargs['box_warp'] # Behind
            all_coordinates = torch.cat([initial_coordinates, perturbed_coordinates], dim=1)
            sigma = self.G.sample_mixed(all_coordinates, torch.randn_like(all_coordinates), ws, update_emas=False)['sigma']
            sigma_initial = sigma[:, :sigma.shape[1]//2]
            sigma_perturbed = sigma[:, sigma.shape[1]//2:]

            monotonic_loss = torch.relu(sigma_initial.detach() - sigma_perturbed).mean() * 10
            monotonic_loss.mul(gain).backward()


            if swapping_prob is not None:
                c_swapped = torch.roll(gen_c.clone(), 1, 0)
                c_gen_conditioning = torch.where(torch.rand([], device=gen_c.device) < swapping_prob, c_swapped, gen_c)
            else:
                c_gen_conditioning = torch.zeros_like(gen_c)

            ws = self.G.mapping(gen_z, c_gen_conditioning, update_emas=False)
            if self.style_mixing_prob > 0:
                with torch.autograd.profiler.record_function('style_mixing'):
                    cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                    cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                    ws[:, cutoff:] = self.G.mapping(torch.randn_like(z), c, update_emas=False)[:, cutoff:]
            initial_coordinates = torch.rand((ws.shape[0], 1000, 3), device=ws.device) * 2 - 1
            perturbed_coordinates = initial_coordinates + torch.randn_like(initial_coordinates) * (1/256) * self.G.rendering_kwargs['box_warp']
            all_coordinates = torch.cat([initial_coordinates, perturbed_coordinates], dim=1)
            sigma = self.G.sample_mixed(all_coordinates, torch.randn_like(all_coordinates), ws, update_emas=False)['sigma']
            sigma_initial = sigma[:, :sigma.shape[1]//2]
            sigma_perturbed = sigma[:, sigma.shape[1]//2:]

            TVloss = torch.nn.functional.l1_loss(sigma_initial, sigma_perturbed) * self.G.rendering_kwargs['density_reg']
            training_stats.report('Loss/G/reg', TVloss)
            TVloss.mul(gain).backward()

        # Alternative density regularization
        if phase in ['Greg', 'Gboth'] and self.G.rendering_kwargs.get('density_reg', 0) > 0 and self.G.rendering_kwargs['reg_type'] == 'monotonic-fixed':
            if swapping_prob is not None:
                c_swapped = torch.roll(gen_c.clone(), 1, 0)
                c_gen_conditioning = torch.where(torch.rand([], device=gen_c.device) < swapping_prob, c_swapped, gen_c)
            else:
                c_gen_conditioning = torch.zeros_like(gen_c)

            ws = self.G.mapping(gen_z, c_gen_conditioning, update_emas=False)

            initial_coordinates = torch.rand((ws.shape[0], 2000, 3), device=ws.device) * 2 - 1 # Front

            perturbed_coordinates = initial_coordinates + torch.tensor([0, 0, -1], device=ws.device) * (1/256) * self.G.rendering_kwargs['box_warp'] # Behind
            all_coordinates = torch.cat([initial_coordinates, perturbed_coordinates], dim=1)
            sigma = self.G.sample_mixed(all_coordinates, torch.randn_like(all_coordinates), ws, update_emas=False)['sigma']
            sigma_initial = sigma[:, :sigma.shape[1]//2]
            sigma_perturbed = sigma[:, sigma.shape[1]//2:]

            monotonic_loss = torch.relu(sigma_initial - sigma_perturbed).mean() * 10
            monotonic_loss.mul(gain).backward()


            if swapping_prob is not None:
                c_swapped = torch.roll(gen_c.clone(), 1, 0)
                c_gen_conditioning = torch.where(torch.rand([], device=gen_c.device) < swapping_prob, c_swapped, gen_c)
            else:
                c_gen_conditioning = torch.zeros_like(gen_c)

            ws = self.G.mapping(gen_z, c_gen_conditioning, update_emas=False)
            if self.style_mixing_prob > 0:
                with torch.autograd.profiler.record_function('style_mixing'):
                    cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                    cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                    ws[:, cutoff:] = self.G.mapping(torch.randn_like(z), c, update_emas=False)[:, cutoff:]
            initial_coordinates = torch.rand((ws.shape[0], 1000, 3), device=ws.device) * 2 - 1
            perturbed_coordinates = initial_coordinates + torch.randn_like(initial_coordinates) * (1/256) * self.G.rendering_kwargs['box_warp']
            all_coordinates = torch.cat([initial_coordinates, perturbed_coordinates], dim=1)
            sigma = self.G.sample_mixed(all_coordinates, torch.randn_like(all_coordinates), ws, update_emas=False)['sigma']
            sigma_initial = sigma[:, :sigma.shape[1]//2]
            sigma_perturbed = sigma[:, sigma.shape[1]//2:]

            TVloss = torch.nn.functional.l1_loss(sigma_initial, sigma_perturbed) * self.G.rendering_kwargs['density_reg']
            training_stats.report('Loss/G/reg', TVloss)
            TVloss.mul(gain).backward()

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if phase in ['Dmain', 'Dboth']:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_img, _gen_ws = self.run_G(gen_z, gen_c, swapping_prob=swapping_prob, neural_rendering_resolution=neural_rendering_resolution, update_emas=True)
                gen_logits = self.run_D(gen_img, gen_c, blur_sigma=blur_sigma, update_emas=True)
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Dgen = torch.nn.functional.softplus(gen_logits)
            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen.mean().mul(gain).backward()

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if phase in ['Dmain', 'Dreg', 'Dboth']:
            name = 'Dreal' if phase == 'Dmain' else 'Dr1' if phase == 'Dreg' else 'Dreal_Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp_image = real_img['image'].detach().requires_grad_(phase in ['Dreg', 'Dboth'])
                real_img_tmp_image_raw = real_img['image_raw'].detach().requires_grad_(phase in ['Dreg', 'Dboth'])
                real_img_tmp = {'image': real_img_tmp_image, 'image_raw': real_img_tmp_image_raw}

                real_logits = self.run_D(real_img_tmp, real_c, blur_sigma=blur_sigma)
                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())

                loss_Dreal = 0
                if phase in ['Dmain', 'Dboth']:
                    loss_Dreal = torch.nn.functional.softplus(-real_logits)
                    training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)

                loss_Dr1 = 0
                if phase in ['Dreg', 'Dboth']:
                    if self.dual_discrimination:
                        with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                            r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp['image'], real_img_tmp['image_raw']], create_graph=True, only_inputs=True)
                            r1_grads_image = r1_grads[0]
                            r1_grads_image_raw = r1_grads[1]
                        r1_penalty = r1_grads_image.square().sum([1,2,3]) + r1_grads_image_raw.square().sum([1,2,3])
                    else: # single discrimination
                        with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                            r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp['image']], create_graph=True, only_inputs=True)
                            r1_grads_image = r1_grads[0]
                        r1_penalty = r1_grads_image.square().sum([1,2,3])
                    loss_Dr1 = r1_penalty * (r1_gamma / 2)
                    training_stats.report('Loss/r1_penalty', r1_penalty)
                    training_stats.report('Loss/D/reg', loss_Dr1)

            with torch.autograd.profiler.record_function(name + '_backward'):
                (loss_Dreal + loss_Dr1).mean().mul(gain).backward()

        if phase in ['D_patchmain', 'D_patchboth']:
            with torch.autograd.profiler.record_function('D_patchmain_forward'):
                gen_img, _gen_ws = self.run_G(gen_z, gen_c, swapping_prob=swapping_prob, neural_rendering_resolution=neural_rendering_resolution, patch_scale=self.patch_scale, run_full=self.dual_patch_discrimination)
                # self.gen_img = gen_img
                _, patch_raw_logits, _ = self.run_D_patch(gen_img, blur_sigma=blur_sigma, feat_match=0, run_patch=False)
                patch_logits, _, _ = self.run_D_patch(real_img, blur_sigma=blur_sigma, feat_match=0, run_patch_raw=False)
                training_stats.report('Loss/scores/patch_fake', patch_raw_logits)
                training_stats.report('Loss/signs/patch_fake', patch_raw_logits.sign())
                training_stats.report('Loss/scores/patch_real', patch_logits)
                training_stats.report('Loss/signs/patch_real', patch_logits.sign())
                loss_D_patch_raw = torch.nn.functional.softplus(patch_raw_logits)
                loss_D_patch = torch.nn.functional.softplus(-patch_logits)
                training_stats.report('Loss/D_patch/loss', loss_D_patch_raw+loss_D_patch)

            with torch.autograd.profiler.record_function('D_patchmain_backward'):
                (loss_D_patch_raw+loss_D_patch).mean().mul(gain).backward()
        
        if phase in ['D_patchreg', 'D_patchboth']:
            with torch.autograd.profiler.record_function('D_patchreg_forward'):
                real_img['patch_image'].requires_grad_(True)
                real_img['patch_image_raw'].requires_grad_(True)
                patch_logits, _, _ = self.run_D_patch(real_img, blur_sigma=blur_sigma, feat_match=0, run_patch_raw=False)
                if self.dual_patch_discrimination:
                    r1_grads = torch.autograd.grad(outputs=[patch_logits.sum()], inputs=[real_img['patch_image'], real_img['patch_image_raw']], create_graph=True, only_inputs=True)
                    r1_grads_image = r1_grads[0]
                    r1_grads_image_raw = r1_grads[1]
                    r1_penalty = r1_grads_image.square().sum([1,2,3]) + r1_grads_image_raw.square().sum([1,2,3])
                else:
                    r1_grads = torch.autograd.grad(outputs=[patch_logits.sum()], inputs=[real_img['patch_image']], create_graph=True, only_inputs=True)
                    r1_grads_image = r1_grads[0]
                    r1_penalty = r1_grads_image.square().sum([1,2,3])
                loss_Dpatch_r1 = r1_penalty * (r1_gamma_patch / 2)
                training_stats.report('Loss/patch_r1_penalty', r1_penalty)
                training_stats.report('Loss/D_patch/reg', loss_Dpatch_r1)

            with torch.autograd.profiler.record_function('D_patchreg_backward'):
                loss_Dpatch_r1.mean().mul(gain).backward()




#----------------------------------------------------------------------------
