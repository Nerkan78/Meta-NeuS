import argparse
import json
import copy
import torch
from torch.cuda import device_count
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets.shapenet import build_shapenet
from models.nerf import build_nerf
from models.rendering import get_rays_shapenet, sample_points, volume_render
from models.NeuSModel import Runner

def NeuS_inner_loop(model, optim, imgs, poses, hwf, bound, num_samples, raybatch_size, inner_steps, scene_idx):
    """
    train the inner model for a specified number of iterations
    """
    pixels = imgs.reshape(-1, 3)

    rays_o, rays_d = get_rays_shapenet(hwf, poses)
    rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)

    num_rays = rays_d.shape[0]
    for step in range(inner_steps):
        indices = torch.randint(num_rays, size=[raybatch_size])
        raybatch_o, raybatch_d = rays_o[indices], rays_d[indices]
        pixelbatch = pixels[indices]
        t_vals, xyz = sample_points(raybatch_o, raybatch_d, bound[0], bound[1],
                                    num_samples, perturb=True)

        near = bound[0]
        far = bound[1]

        use_white_bkgd = True
        background_rgb = None
        if use_white_bkgd:
            background_rgb = torch.ones([1, 3]).to('cuda')
        optim.zero_grad()
        render_out = model.renderer.render(
            raybatch_o, raybatch_d, near, far, scene_idx,
            background_rgb=background_rgb,
            cos_anneal_ratio=model.get_cos_anneal_ratio(),
            compute_eikonal_loss=model.igr_weight > 0,
            compute_radiance_grad_loss=model.radiance_grad_weight > 0)

        color_fine = render_out['color_fine']
        # print(f'--------COLOR FINE SIZE {color_fine.size()}')
        # rgbs, sigmas = model(xyz)
        # colors = volume_render(rgbs, sigmas, t_vals, white_bkgd=True)
        # loss = F.mse_loss(colors, pixelbatch)
        # loss.backward()

        loss = 0

        # Image loss: L1
        target_rgb = pixelbatch
        # print(f'--------target_rgb SIZE {target_rgb.size()}')

        color_fine_loss = (color_fine - target_rgb).abs().mean()
        loss += color_fine_loss

        # psnr_train = psnr(color_fine, target_rgb)

        # SDF loss: eikonal
        if model.igr_weight > 0:
            # boolean mask, 1 if point is inside 1-sphere
            relax_inside_sphere = render_out['relax_inside_sphere']

            gradients_eikonal = render_out['gradients_eikonal']
            gradient_error = (torch.linalg.norm(gradients_eikonal, ord=2, dim=-1) - 1.0) ** 2
            eikonal_loss = \
                (relax_inside_sphere * gradient_error).sum() / \
                (relax_inside_sphere.sum() + 1e-5)
            loss += eikonal_loss * model.igr_weight

        # Radiance loss: gradient orthogonality
        if model.radiance_grad_weight > 0:
            # dim 0: gradient of r/g/b
            # dim 1: point number
            # dim 2: gradient over x/y/z
            gradients_radiance = render_out['gradients_radiance']  # 3, K, 3
            gradients_eikonal = render_out['gradients_eikonal']

            gradients_eikonal_ = gradients_eikonal[None]  # 1, K, 3

            # We want these gradients to be orthogonal, so force dot product to zero
            grads_dot_product = (gradients_eikonal * gradients_radiance).sum(-1)  # 3, K
            radiance_grad_loss = (grads_dot_product ** 2).mean()
            # radiance_grad_loss = torch.nn.HuberLoss(delta=0.0122)(
            #     grads_dot_product, ZERO.expand_as(grads_dot_product))
            loss += radiance_grad_loss * model.radiance_grad_weight

        optim.step()


def inner_loop(model, optim, imgs, poses, hwf, bound, num_samples, raybatch_size, inner_steps):
    """
    train the inner model for a specified number of iterations
    """
    pixels = imgs.reshape(-1, 3)

    rays_o, rays_d = get_rays_shapenet(hwf, poses)
    rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)

    num_rays = rays_d.shape[0]
    for step in range(inner_steps):
        indices = torch.randint(num_rays, size=[raybatch_size])
        raybatch_o, raybatch_d = rays_o[indices], rays_d[indices]
        pixelbatch = pixels[indices] 
        t_vals, xyz = sample_points(raybatch_o, raybatch_d, bound[0], bound[1],
                                    num_samples, perturb=True)
        
        optim.zero_grad()
        rgbs, sigmas = model(xyz)
        colors = volume_render(rgbs, sigmas, t_vals, white_bkgd=True)
        loss = F.mse_loss(colors, pixelbatch)
        loss.backward()
        optim.step()


def train_meta(args, meta_model, meta_optim, data_loader, device):
    """
    train the meta_model for one epoch using reptile meta learning
    https://arxiv.org/abs/1803.02999
    """
    for imgs, poses, hwf, bound in data_loader:
        imgs, poses, hwf, bound = imgs.to(device), poses.to(device), hwf.to(device), bound.to(device)
        imgs, poses, hwf, bound = imgs.squeeze(), poses.squeeze(), hwf.squeeze(), bound.squeeze()
        scene_idx = 0
        meta_optim.zero_grad()

        inner_model = copy.deepcopy(meta_model)
        inner_optim = torch.optim.SGD(inner_model.renderer.parameters(), args.inner_lr)

        NeuS_inner_loop(inner_model, inner_optim, imgs, poses,
                    hwf, bound, args.num_samples,
                    args.train_batchsize, args.inner_steps, scene_idx)
        
        with torch.no_grad():
            for meta_param, inner_param in zip(meta_model.renderer.parameters(), inner_model.renderer.parameters()):
                meta_param.grad = meta_param - inner_param
        
        meta_optim.step()


def report_result(model, imgs, poses, hwf, bound, num_samples, raybatch_size):
    """
    report view-synthesis result on heldout views
    """
    ray_origins, ray_directions = get_rays_shapenet(hwf, poses)

    view_psnrs = []
    for img, rays_o, rays_d in zip(imgs, ray_origins, ray_directions):
        rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)
        t_vals, xyz = sample_points(rays_o, rays_d, bound[0], bound[1],
                                    num_samples, perturb=False)
        
        synth = []
        num_rays = rays_d.shape[0]
        with torch.no_grad():
            for i in range(0, num_rays, raybatch_size):
                rgbs_batch, sigmas_batch = model(xyz[i:i+raybatch_size])
                color_batch = volume_render(rgbs_batch, sigmas_batch, 
                                            t_vals[i:i+raybatch_size],
                                            white_bkgd=True)
                synth.append(color_batch)
            synth = torch.cat(synth, dim=0).reshape_as(img)
            error = F.mse_loss(img, synth)
            psnr = -10*torch.log10(error)
            view_psnrs.append(psnr)
    
    scene_psnr = torch.stack(view_psnrs).mean()
    return scene_psnr


def val_meta(args, model, val_loader, device):
    """
    validate the meta trained model for few-shot view synthesis
    """
    meta_trained_state = model.renderer.state_dict()
    val_model = copy.deepcopy(model)
    
    val_psnrs = []
    for imgs, poses, hwf, bound in val_loader:
        imgs, poses, hwf, bound = imgs.to(device), poses.to(device), hwf.to(device), bound.to(device)
        imgs, poses, hwf, bound = imgs.squeeze(), poses.squeeze(), hwf.squeeze(), bound.squeeze()
        scene_idx = 0

        tto_imgs, test_imgs = torch.split(imgs, [args.tto_views, args.test_views], dim=0)
        tto_poses, test_poses = torch.split(poses, [args.tto_views, args.test_views], dim=0)

        val_model.renderer.load_state_dict(meta_trained_state)
        val_optim = torch.optim.SGD(val_model.renderer.parameters(), args.tto_lr)

        NeuS_inner_loop(val_model, val_optim, tto_imgs, tto_poses, hwf,
                    bound, args.num_samples, args.tto_batchsize, args.tto_steps, scene_idx)
        
        scene_psnr = report_result(val_model, test_imgs, test_poses, hwf, bound, 
                                    args.num_samples, args.test_batchsize)
        val_psnrs.append(scene_psnr)

    val_psnr = torch.stack(val_psnrs).mean()
    return val_psnr


def main():
    parser = argparse.ArgumentParser(description='shapenet few-shot view synthesis')
    parser.add_argument('--config', type=str, required=True,
                        help='config file for the shape class (cars, chairs or lamps)')
    parser.add_argument('--neus_conf', type=str, required=True,
                        help='config filepath for the NeuS renderer')
    parser.add_argument('--device', type=str, required=True,
                        help='computational device')
    args = parser.parse_args()

    with open(args.config) as config:
        info = json.load(config)
        for key, value in info.items():
            args.__dict__[key] = value

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_set = build_shapenet(image_set="train", dataset_root=args.dataset_root,
                                splits_path=args.splits_path, num_views=args.train_views)
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True)

    val_set = build_shapenet(image_set="val", dataset_root=args.dataset_root,
                            splits_path=args.splits_path,
                            num_views=args.tto_views+args.test_views)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

    # meta_model = build_nerf(args)
    meta_model = Runner(args, device)
    # meta_model.to(device)

    meta_optim = torch.optim.Adam(meta_model.renderer.parameters(), lr=args.meta_lr)

    for epoch in range(1, args.meta_epochs+1):
        train_meta(args, meta_model, meta_optim, train_loader, device)
        val_psnr = val_meta(args, meta_model, val_loader, device)
        print(f"Epoch: {epoch}, val psnr: {val_psnr:0.3f}")

        torch.save({
            'epoch': epoch,
            'meta_model_state_dict': meta_model.renderer.state_dict(),
            'meta_optim_state_dict': meta_optim.state_dict(),
            }, f'meta_epoch{epoch}.pth')


if __name__ == '__main__':
    main()