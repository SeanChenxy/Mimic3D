import os
import numpy as np
import dnnlib
import copy
import legacy
import scipy
import imageio
from tqdm import tqdm
import torch
from torch_utils import misc
import cv2

from training.training_loop import save_image_grid
from gen_videos import layout_grid, create_samples
from camera_utils import LookAtPoseSampler, UniformCameraPoseSampler, FOV_to_intrinsics
from shape_utils import convert_sdf_samples_to_ply
import trimesh
import pyrender
import PIL.Image
from sklearn import decomposition

os.environ['PYOPENGL_PLATFORM'] = 'osmesa'


def visualize(mesh, cam_pose=None):
    scene = pyrender.Scene(ambient_light=[.1, 0.1, 0.1], bg_color=[1.0, 1.0, 1.0])
    # cam = pyrender.PerspectiveCamera(yfov=np.deg2rad(60), aspectRatio=1)
    cam = pyrender.OrthographicCamera(xmag=0.3, ymag=0.3)
    node_cam = pyrender.Node(camera=cam, matrix=np.eye(4))
    scene.add_node(node_cam)
    cam_pose = np.array(
        [[-1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, -1, -0.5],
        [0, 0, 0, 1],], dtype='float32')
    scene.set_pose(node_cam, pose=cam_pose)
    light = pyrender.DirectionalLight(color=[1, 1, 1], intensity=1500)
    scene.add(light, pose=cam_pose)


    # update scene
    scene.add(pyrender.Mesh.from_trimesh(mesh))
    viewer = pyrender.OffscreenRenderer(512, 512)

    color, depth = viewer.render(scene)
    cv2.imwrite(f'tmp/test.png', color[..., ::-1])
    return color[..., ::-1]

def inference(
        opts                    = None,
        run_dir                 = '.',      # Output directory.
        training_set_kwargs     = {},       # Options for training set.
        G_kwargs                = {},       # Options for generator network.
        random_seed             = 0,        # Global random seed.
        num_gpus                = 1,        # Number of GPUs participating in the training.
        rank                    = 0,        # Rank of the current process in [0, num_gpus[.
        batch_size              = 4,        # Total batch size for one training iteration. Can be larger than batch_gpu * num_gpus.
        resume_pkl              = None,     # Network pickle to resume training from.
        cudnn_benchmark         = True,     # Enable torch.backends.cudnn.benchmark?
        **_
    ):

    
    device = torch.device('cuda', rank)
    np.random.seed(random_seed * num_gpus + rank)
    torch.manual_seed(random_seed * num_gpus + rank)
    torch.backends.cudnn.benchmark = cudnn_benchmark    # Improves training speed.
    torch.backends.cuda.matmul.allow_tf32 = False       # Improves numerical accuracy.
    torch.backends.cudnn.allow_tf32 = False             # Improves numerical accuracy.
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False  # Improves numerical accuracy.

    # Load training set.
    if rank == 0:
        print('Loading training set...')
    training_set = dnnlib.util.construct_class_by_name(**training_set_kwargs) # subclass of training.dataset.Dataset
    if rank == 0:
        print()
        print('Num images: ', len(training_set))
        print('Image shape:', training_set.image_shape)
        print('Label shape:', training_set.label_shape)
        print()
        
    # Construct networks.
    if rank == 0:
        print('Constructing networks...')
    common_kwargs = dict(c_dim=training_set.label_dim, img_resolution=training_set.resolution, img_channels=training_set.num_channels)
    G = dnnlib.util.construct_class_by_name(**G_kwargs, **common_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
    G.register_buffer('dataset_label_std', torch.tensor(training_set.get_label_std()).to(device))
    G_ema = copy.deepcopy(G).eval()

    # Resume from existing pickle.
    if (resume_pkl is not None) and (rank == 0):
        if 'pkl' not in resume_pkl:
            resume_pkl = os.path.join(run_dir, f'network-snapshot-{resume_pkl}.pkl')
        print(f'Resuming from "{resume_pkl}"')
        with dnnlib.util.open_url(resume_pkl) as f:
            resume_data = legacy.load_network_pkl(f)
        for name, module in [('G_ema', G_ema)]:
            misc.copy_params_and_buffers(resume_data[name], module, require_all=False)
    else:
        print(f'w/o resume !!!')
    print()
    
    model_name = resume_pkl.split('-')[-1].split('.')[0] if resume_pkl is not None else 'test'
    os.makedirs(os.path.join(run_dir, f'infer/{model_name}' ), exist_ok=True)
    focal_length = 4.2647 if opts.cfg.lower() != 'shapenet' else 1.7074 # shapenet has higher FOV
    intrinsics = torch.tensor([[focal_length, 0, 0.5], [0, focal_length, 0.5], [0, 0, 1]], device=device)
    camera_lookat_point = torch.tensor(G_ema.rendering_kwargs['avg_camera_pivot'], device=device)
    seeds = opts.seeds
    if len(seeds) == 2:
        seeds = list(range(seeds[0], seeds[1]))
    if opts.inference_mode == 'image':
        os.makedirs(os.path.join(run_dir, f'infer/{model_name}/images' ), exist_ok=True)
        # Setup z,c.
        gen_z = torch.from_numpy(np.stack([np.random.RandomState(seed).randn(G.z_dim) for seed in seeds])).to(device)
        gen_c = torch.cat([
            LookAtPoseSampler.sample(
                3.14/2 + opts.yaw_range,
                3.14/2 -0.05 + opts.pitch_range,
                # 3.14/2,
                # 3.14/2,
                camera_lookat_point,
                radius=G_ema.rendering_kwargs['avg_camera_radius'], device=device).reshape(-1, 16), 
                intrinsics.reshape(-1, 9)], 1).repeat(len(seeds), 1)

        if rank == 0:
            print(f'Render images under {opts.neural_rendering_resolution_infer} resolution ...')
        ws = G_ema.mapping(z=gen_z, c=gen_c, truncation_psi=opts.truncation_psi, truncation_cutoff=opts.truncation_cutoff)
        gen_img_list = []
        if opts.retplane<0:
            for i, (w, c) in enumerate(zip(torch.split(ws, 2, dim=0), torch.split(gen_c, 2, dim=0))):
                gen_img = G_ema.synthesis(w, c, neural_rendering_resolution=opts.neural_rendering_resolution_infer, patch_scale=(1., opts.patch_scale)['patch' in opts.image_mode], chunk=(None, opts.chunk//opts.batch*opts.gpus)[opts.chunk>0], noise_mode='const', ret_plane=opts.retplane>=0, coarse=opts.get('coarse', True))
                gen_img_list.append(gen_img)
                print(f'infer {(i+1)*2} samples')
            gen_img = {k: torch.cat([o[k] for o in gen_img_list], 0) for k,v in gen_img.items()}
        else:
            gen_img = G_ema.synthesis(ws, gen_c, neural_rendering_resolution=opts.neural_rendering_resolution_infer, patch_scale=(1., opts.patch_scale)['patch' in opts.image_mode], chunk=(None, opts.chunk//opts.batch*opts.gpus)[opts.chunk>0], noise_mode='const', ret_plane=opts.retplane>=0, coarse=opts.get('coarse', True))

        if opts.neural_rendering_resolution_infer != opts.resize_resolution:
            images_raw = torch.nn.functional.interpolate(gen_img[opts.image_mode], size=(opts.resize_resolution),
                        mode='bilinear', align_corners=False, antialias=True).cpu().numpy()
        else:
            images_raw = gen_img[opts.image_mode].cpu().numpy()
        if opts.retplane>=0:
            plane = gen_img['plane']
            if isinstance(plane, list):
                if opts.retplane < len(plane):
                    plane = plane[opts.retplane]
                    plane = plane.view(len(seeds)*3, plane.shape[-3], plane.shape[-2], plane.shape[-1])
                    plane = torch.nn.functional.interpolate(plane, size=(opts.resize_resolution),
                            mode='bilinear', align_corners=False, antialias=True).view(len(seeds), 3, plane.shape[-3], opts.resize_resolution, opts.resize_resolution).cpu().numpy()
                else:
                    plane0 = plane[0].view(len(seeds)*3, plane[0].shape[-3], plane[0].shape[-2], plane[0].shape[-1])
                    plane0 = torch.nn.functional.interpolate(plane0, size=(opts.resize_resolution),
                            mode='bilinear', align_corners=False, antialias=True).view(len(seeds), 3, plane[0].shape[-3], opts.resize_resolution, opts.resize_resolution).cpu().numpy()
                    plane1 = plane[1].view(len(seeds)*3, plane[1].shape[-3], plane[1].shape[-2], plane[1].shape[-1])
                    plane1 = torch.nn.functional.interpolate(plane1, size=(opts.resize_resolution),
                            mode='bilinear', align_corners=False, antialias=True).view(len(seeds), 3, plane[1].shape[-3], opts.resize_resolution, opts.resize_resolution).cpu().numpy()
                    plane = plane0 + plane1
            else:
                plane = plane.view(len(seeds)*3, plane.shape[-3], plane.shape[-2], plane.shape[-1])
                plane = torch.nn.functional.interpolate(plane, size=(opts.resize_resolution),
                            mode='bilinear', align_corners=False, antialias=True).view(len(seeds), 3, plane.shape[-3], opts.resize_resolution, opts.resize_resolution).cpu().numpy()
            plane = plane[:, :, :, ::-1]
            plane[:, 1] = plane[:, 1].transpose(0,1,3,2)
            plane[:, 1] = plane[:, 1, :, ::-1, ::-1]
            plane = np.linalg.norm(plane, axis=2, ord=2, keepdims=True)
            plane[:, 1:] *= -1
            plane_min = plane.min(axis=-1, keepdims=True).min(axis=-2, keepdims=True)
            plane_max = plane.max(axis=-1, keepdims=True).max(axis=-2, keepdims=True)
            plane = (255 * (plane-plane_min) / (plane_max-plane_min)).clip(0, 255).astype(np.uint8)
            plane = np.concatenate([plane[:, i] for i in range(plane.shape[1])], -1)

            plane = np.concatenate( [cv2.applyColorMap(plane[i, 0], 12)[None] for i in range(plane.shape[0])], 0)
            plane = plane.transpose(0, 3, 1, 2).astype('float32') / 128.0 - 1.0
            plane = plane[:, [2,1,0]]
            images_raw = np.concatenate([images_raw, plane], -1)
        if opts.shapes:
            print('Generating shape')
            voxel_resolution = 512
            max_batch = 1000000
            head = 0
            samples, voxel_origin, voxel_size = create_samples(N=voxel_resolution, voxel_origin=[0, 0, 0], cube_length=G_ema.rendering_kwargs['box_warp'])
            samples = samples.repeat(ws.shape[0],1,1).to(device)
            sigmas = torch.zeros((samples.shape[0], samples.shape[1], 1), device=device)
            transformed_ray_directions_expanded = torch.zeros((samples.shape[0], max_batch, 3), device=device)
            transformed_ray_directions_expanded[..., -1] = -1
            with torch.no_grad():
                while head < samples.shape[1]:
                    torch.manual_seed(0)
                    sigma = G_ema.sample_mixed(samples[:, head:head+max_batch], transformed_ray_directions_expanded[:, :samples.shape[1]-head], ws, truncation_psi=opts.truncation_psi, truncation_cutoff=opts.truncation_cutoff, noise_mode='const')['sigma']
                    sigmas[:, head:head+max_batch] = sigma
                    head += max_batch
            sigmas = sigmas.reshape((ws.shape[0], voxel_resolution, voxel_resolution, voxel_resolution)).cpu().numpy()
            sigmas = np.flip(sigmas, 1)
            pad = int(30 * voxel_resolution / 256)
            pad_top = int(38 * voxel_resolution / 256)
            sigmas[:, :pad] = 0
            sigmas[:, -pad:] = 0
            sigmas[:, :, :pad] = 0
            sigmas[:, :, -pad_top:] = 0
            sigmas[:, :, :, :pad] = 0
            sigmas[:, :, :, -pad:] = 0
            
            shape_list = []
            for i, s in enumerate(seeds):
                verts, faces = convert_sdf_samples_to_ply(sigmas[i], voxel_origin, voxel_size, os.path.join(run_dir, f'infer/{model_name}/shape{s}.ply'), level=10)
                roty = np.array([[np.cos(opts.yaw_range), 0, np.sin(opts.yaw_range)],
                                [0,1,0],
                                [-np.sin(opts.yaw_range), 0, np.cos(opts.yaw_range)]])
                verts = np.dot(roty, verts.T).T
                mesh = trimesh.Trimesh(verts, faces)
                img = visualize(mesh).astype('float32') / 128.0 - 1.0
                shape_list.append(cv2.resize(img, (opts.resize_resolution, opts.resize_resolution))[None])
            shape = np.concatenate(shape_list, 0).transpose(0, 3, 1, 2)
            images_raw = np.concatenate([images_raw, shape], -1)
        save_image_grid(
            images_raw * (1, -1)['depth' in opts.image_mode], 
            os.path.join(run_dir, f'infer/{model_name}/images', f'r{opts.neural_rendering_resolution_infer}_{opts.yaw_range}_{opts.pitch_range}_{opts.image_mode}_r{opts.retplane}.png'), 
            drange=([-1, 1], [-images_raw.max(), -images_raw.min()])['depth' in opts.image_mode], 
            grid_size=opts.grid,
            seeds=seeds)
    elif opts.inference_mode == 'video':
        grid_w = opts.grid[0]
        grid_h = opts.grid[1]
        num_keyframes = len(seeds) // (grid_w*grid_h)

        all_seeds = np.zeros(num_keyframes*grid_h*grid_w, dtype=np.int64)
        for idx in range(num_keyframes*grid_h*grid_w):
            all_seeds[idx] = seeds[idx % len(seeds)]

        gen_z = torch.from_numpy(np.stack([np.random.RandomState(seed).randn(G.z_dim) for seed in all_seeds])).to(device)
        cam2world_pose = LookAtPoseSampler.sample(3.14/2, 3.14/2, camera_lookat_point, radius=G_ema.rendering_kwargs['avg_camera_radius'], device=device)
        c = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
        c = c.repeat(len(gen_z), 1)
        ws = G_ema.mapping(z=gen_z, c=c, truncation_psi=opts.truncation_psi, truncation_cutoff=opts.truncation_cutoff)
        _ = G_ema.synthesis(ws[:1], c[:1]) # warm up
        ws = ws.reshape(grid_h, grid_w, num_keyframes, *ws.shape[1:])

        # Interpolation.
        grid = []
        for yi in range(grid_h):
            row = []
            for xi in range(grid_w):
                x = np.arange(-num_keyframes * 2, num_keyframes * (2 + 1))
                y = np.tile(ws[yi][xi].cpu().numpy(), [2 * 2 + 1, 1, 1])
                interp = scipy.interpolate.interp1d(x, y, kind='cubic', axis=0)
                row.append(interp)
            grid.append(row)

        # Render video.
        video_path = os.path.join(run_dir, f'infer/{model_name}', f'seed{opts.seeds}_res{opts.neural_rendering_resolution_infer}_{opts.yaw_range}_{opts.pitch_range}_{opts.image_mode}.mp4')

        print(f'Video save at {video_path}')
        video_out = imageio.get_writer(video_path, mode='I', fps=60, codec='libx264', bitrate='10M')
        all_poses = []
        for frame_idx in tqdm(range(num_keyframes * opts.w_frames)):
            imgs = []
            for yi in range(grid_h):
                for xi in range(grid_w):
                    cam2world_pose = LookAtPoseSampler.sample(3.14/2 + opts.yaw_range * np.sin(2 * 3.14 * frame_idx / (num_keyframes * opts.w_frames)),
                                                            3.14/2 -0.05 + opts.pitch_range * np.cos(2 * 3.14 * frame_idx / (num_keyframes * opts.w_frames)),
                                                            camera_lookat_point, radius=G_ema.rendering_kwargs['avg_camera_radius'], device=device)
                    all_poses.append(cam2world_pose.squeeze().cpu().numpy())
                    focal_length = 4.2647 if opts.cfg.lower() != 'shapenet' else 1.7074 # shapenet has higher FOV
                    intrinsics = torch.tensor([[focal_length, 0, 0.5], [0, focal_length, 0.5], [0, 0, 1]], device=device)
                    c = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

                    interp = grid[yi][xi]
                    w = torch.from_numpy(interp(frame_idx / opts.w_frames)).to(device)

                    entangle = 'camera'
                    if entangle == 'conditioning':
                        c_forward = torch.cat([LookAtPoseSampler.sample(3.14/2,
                                                                        3.14/2,
                                                                        camera_lookat_point,
                                                                        radius=G.rendering_kwargs['avg_camera_radius'], device=device).reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
                        w_c = G_ema.mapping(z=zs[0:1], c=c[0:1], truncation_psi=opts.truncation_psi, truncation_cutoff=opts.truncation_cutoff)
                        img = G_ema.synthesis(ws=w_c, c=c_forward, neural_rendering_resolution=opts.neural_rendering_resolution_infer, noise_mode='const', coarse=opts.get('coarse', True), chunk=(None, opts.chunk)[opts.chunk>0])[opts.image_mode][0]
                    elif entangle == 'camera':
                        img = G_ema.synthesis(ws=w.unsqueeze(0), c=c[0:1], neural_rendering_resolution=opts.neural_rendering_resolution_infer, noise_mode='const', coarse=opts.get('coarse', True), chunk=(None, opts.chunk)[opts.chunk>0])[opts.image_mode][0]
                    elif entangle == 'both':
                        w_c = G_ema.mapping(z=zs[0:1], c=c[0:1], truncation_psi=opts.truncation_psi, truncation_cutoff=opts.truncation_cutoff)
                        img = G_ema.synthesis(ws=w_c, c=c[0:1], neural_rendering_resolution=opts.neural_rendering_resolution_infer, noise_mode='const', coarse=opts.get('coarse', True), chunk=(None, opts.chunk)[opts.chunk>0])[opts.image_mode][0]

                    if opts.image_mode == 'image_depth':
                        img = -img
                        img = (img - img.min()) / (img.max() - img.min()) * 2 - 1
                    
                    if img.shape[-1] != opts.resize_resolution:
                        img = torch.nn.functional.interpolate(img[None], size=(opts.resize_resolution),
                                mode='bilinear', align_corners=False, antialias=True)[0]
                    imgs.append(img)
            video_out.append_data(layout_grid(torch.stack(imgs), grid_w=grid_w, grid_h=grid_h))
        video_out.close()
    else:
        all_seeds = np.zeros(len(seeds), dtype=np.int64)
        for idx in range(len(seeds)):
            all_seeds[idx] = seeds[idx % len(seeds)]
        gen_z = torch.from_numpy(np.stack([np.random.RandomState(seed).randn(G.z_dim) for seed in all_seeds])).to(device)
        cam2world_pose = LookAtPoseSampler.sample(3.14/2, 3.14/2, camera_lookat_point, radius=G_ema.rendering_kwargs['avg_camera_radius'], device=device)
        c = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
        c = c.repeat(len(gen_z), 1)
        ws = G_ema.mapping(z=gen_z, c=c, truncation_psi=opts.truncation_psi, truncation_cutoff=opts.truncation_cutoff)
        _ = G_ema.synthesis(ws[:1], c[:1]) # warm up

        for s_, w in zip(all_seeds, ws):
            # Render video.
            video_path = os.path.join(run_dir, f'infer/{model_name}', f'seed{s_}_res{opts.neural_rendering_resolution_infer}_{opts.yaw_range}_{opts.pitch_range}_{opts.image_mode}.mp4')
            save_img_dir =  video_path[:-4]
            os.makedirs(save_img_dir, exist_ok=True)
            print(f'Video save at {video_path}')
            video_out = imageio.get_writer(video_path, mode='I', fps=60, codec='libx264', bitrate='10M')
            for frame_idx in tqdm(range(opts.w_frames)):
                imgs = []
                cam2world_pose = LookAtPoseSampler.sample(3.14/2 + opts.yaw_range * np.sin(2 * 3.14 * frame_idx / opts.w_frames),
                                                            3.14/2 -0.05 + opts.pitch_range * np.cos(2 * 3.14 * frame_idx / opts.w_frames),
                                                            camera_lookat_point, radius=G_ema.rendering_kwargs['avg_camera_radius'], device=device)
                focal_length = 4.2647 if opts.cfg.lower() != 'shapenet' else 1.7074 # shapenet has higher FOV
                intrinsics = torch.tensor([[focal_length, 0, 0.5], [0, focal_length, 0.5], [0, 0, 1]], device=device)
                c = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
                img = G_ema.synthesis(ws=w.unsqueeze(0), c=c[0:1], neural_rendering_resolution=opts.neural_rendering_resolution_infer, noise_mode='const', coarse=opts.get('coarse', True), chunk=(None, opts.chunk)[opts.chunk>0])[opts.image_mode][0]
                if img.shape[-1] != opts.resize_resolution:
                    img = torch.nn.functional.interpolate(img[None], size=(opts.resize_resolution), mode='bilinear', align_corners=False, antialias=True)[0]
                imgs.append(img)
                img_save = (img * 127.5 + 128).permute(1,2,0).clamp(0, 255).to(torch.uint8).cpu().numpy()
                PIL.Image.fromarray(img_save, 'RGB').save(os.path.join(save_img_dir, f'{frame_idx}.png'))
                video_out.append_data(layout_grid(torch.stack(imgs), grid_w=1, grid_h=1))
        video_out.close()
