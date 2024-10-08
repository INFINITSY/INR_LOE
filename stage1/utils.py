import copy
import logging
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import tqdm
from sklearn.metrics import recall_score
from skimage.metrics import structural_similarity as ssim

from stage1.nerf.helper import *


def get_mgrid(sidelen, dim=2, max=1.0, sidepatch=1, padding_ratio=0.):
    """Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int,
    sidepatch: int. Number of patches along each side of the image.
    padding_ratio: float. The ratio of padding points to add to the grid.
    """
    assert sidelen % sidepatch == 0, "sidelen must be divisible by npatch"
    patchlen = sidelen // sidepatch

    if padding_ratio > 0:
        # Calculate the number of padding points based on the ratio
        num_padding = int(padding_ratio * patchlen)
        # Extend the grid to include padding
        max += (2 * max / (patchlen - 1)) * num_padding
        patchlen += 2 * num_padding

    tensors = tuple(dim * [torch.linspace(-max, max, steps=patchlen)])
    mgrid = torch.stack(torch.meshgrid(*tensors, indexing="ij"), dim=-1)
    mgrid = mgrid.reshape(-1, dim) # (patchlen^dim, dim)

    mgrid = mgrid.unsqueeze(0).repeat(sidepatch ** dim, 1, 1) 
    # (sidepatch^dim, patchlen^dim, dim)

    return mgrid


def get_load(gate):
    """Compute the true load per expert, given the gates.
    The load is the number of examples for which the corresponding gate is >0.
    Args:
        gates: a `Tensor` of shape [batch_size, n]

    Returns:
        a float32 `Tensor` of shape [n]
    """
    return (gate > 0).sum(0)


def get_sparsity(gate, l1_exp=1.0):
    # L1 penalty on the gates encourages sparsity
    weights = 1. / torch.pow(torch.full((gate.shape[-1],), l1_exp, device=gate.device), \
                torch.arange(gate.shape[-1], device=gate.device))
    sparsity = torch.mean(torch.abs(gate) * weights[None, ...])
    return sparsity


def get_masks(side_length, side_patch, padding_ratio=0.):
    '''
    Generate masks for the patches, psnr calculation, and overlap
    Args:
        side_length: the side length of the image
        side_patch: the number of patches along each side of the image
        padding_ratio: the ratio of padding to the patch size

    H', W' = npatch * (patch_size + 2 * padding)

    Returns:
        mask_all: mask for all the pixels except the border padding. (H', W')
        mask_psnr: mask for psnr calculation. (H', W')
        mask_overlap: mask for overlap. (4, H', W')
    '''
    patch_size = side_length // side_patch
    npatch = side_length // patch_size
    padding = int(padding_ratio * patch_size)
    padded_patch = patch_size + 2 * padding
    H = W = npatch * padded_patch
    
    # base mask of shape (npatch, npatch, padded_patch, padded_patch)
    mask = torch.zeros(npatch, npatch, padded_patch, padded_patch, dtype=torch.bool)

    def reshape_and_permute(m):
        return m.permute(0, 2, 1, 3).reshape(H, W)

    # mask for psnr calculation
    mask_psnr = mask.clone()
    mask_psnr[..., padding:-padding, padding:-padding] = 1
    mask_psnr = reshape_and_permute(mask_psnr)
    
    # mask for all the pixels except the border padding
    mask_all = mask.clone()
    mask_all = reshape_and_permute(mask_all)
    mask_all[padding:-padding, padding:-padding] = 1

    # mask for overlap
    # right overlap
    mask_r = mask.clone()
    mask_r[..., :, patch_size:] = 1
    mask_r = reshape_and_permute(mask_r)
    mask_r[:, -2 * padding:] = 0

    # left overlap
    mask_l = mask.clone()
    mask_l[..., :, :-patch_size] = 1
    mask_l = reshape_and_permute(mask_l)
    mask_l[:, :2 * padding] = 0

    # bottom overlap
    mask_b = mask.clone()
    mask_b[..., patch_size:, :] = 1
    mask_b = reshape_and_permute(mask_b)
    mask_b[-2 * padding:, :] = 0

    # top overlap
    mask_t = mask.clone()
    mask_t[..., :-patch_size, :] = 1
    mask_t = reshape_and_permute(mask_t)
    mask_t[:2 * padding, :] = 0

    # stack the masks
    mask_overlap = torch.stack([mask_r, mask_l, mask_b, mask_t], dim=0) # shape: (4, H, W)
    # remove borader
    border_mask = torch.ones_like(mask_overlap, dtype=torch.bool)
    border_mask[:, padding:-padding, padding:-padding] = 0
    mask_overlap[border_mask] = 0
    mask_overlap = mask_overlap

    return mask_all.reshape(-1), mask_psnr.reshape(-1), mask_overlap.reshape(4, -1)


def patchify_with_padding(img, side_patch, padding_ratio=0.):
    """
    Split the image into patches. Allow for padding.
    """
    N, C, H, W = img.shape
    patch_size = H // side_patch
    stride = patch_size
    npatch = H // patch_size

    # pad the image
    padding = int(padding_ratio * patch_size)
    pad = (padding, padding, padding, padding)
    padded_img = F.pad(img, pad, mode="constant", value=0) # (N, C, H+2*padding, W+2*padding)
    padded_patch = patch_size + 2 * padding

    # unfold the image
    patches = padded_img.unfold(2, padded_patch, stride).unfold(3, padded_patch, stride)
    # shape is (N, C, npatch, npatch, padded_patch, padded_patch)

    patches = patches.permute(0, 1, 2, 4, 3, 5)
    patches = patches.reshape(N, C, npatch * padded_patch, npatch * padded_patch)
    # shape is (N, C, H', W')

    return patches


def compute_loss(args, epoch, out, y, criterion, gates=None, importance=None, top_k=False, 
                 mask_overlap=None, mask_psnr=None, mask_all=None):
    # base mse loss
    if mask_psnr is not None:
        mask_psnr = mask_psnr.unsqueeze(0).unsqueeze(-1).expand_as(y)
    if mask_all is not None:
        # mask_all shape: H*W. Need to reshape to (N, H*W, C) as y
        mask_all = mask_all.unsqueeze(0).unsqueeze(-1).expand_as(y)
        mse = criterion(out[mask_all], y[mask_all])
    else:
        mse = criterion(out, y)
    loss = mse

    # overlap consistency loss
    if mask_overlap is not None:
        mask_overlap = mask_overlap.unsqueeze(1).unsqueeze(-1).expand(
            -1, y.size(0), -1, y.size(-1)
        )
        loss += args.overlap_loss_w * (
            criterion(out[mask_overlap[0]], out[mask_overlap[1]]) + 
            criterion(out[mask_overlap[2]], out[mask_overlap[3]])
        )

    # sparsity loss
    if args.sparse_loss_w > 0 and top_k is False:
        sparse_loss = []
        for l, g in enumerate(gates):
            # compute the sparse loss
            sparse_loss.append(get_sparsity(g))
            if args.progressive_epoch is not None and l == epoch // args.progressive_epoch:
                # only calculate the sparse loss for active layers
                break
        sparse_loss = torch.stack(sparse_loss).mean()
        loss += args.sparse_loss_w * sparse_loss

    # cv loss
    if args.cv_loss_w > 0:
        cv_loss_imp = []
        cv_loss_load = []
        for l, imp in enumerate(importance):
            cv_loss_imp.append(get_cv_loss(imp))
            if top_k:
                load = get_load(gates[l])
                cv_loss_load.append(get_cv_loss(load))
            if args.progressive_epoch is not None and l == epoch // args.progressive_epoch:
                # only calculate the cv loss for active layers
                break
        cv_loss_imp = torch.stack(cv_loss_imp).mean()
        cv_loss_load = torch.stack(cv_loss_load).mean() if top_k else 0
        loss += args.cv_loss_w * (cv_loss_imp + cv_loss_load)

    if args.std_loss_w > 0:
        std_loss = []
        # gates is a list: num_layers x [N, num_experts]
        # std_threshold = [0.005, 0.01, 0.05, 0.1, 0.15]
        std_threshold = [0.1] * len(gates)
        for l, g in enumerate(gates):
            # compute the variance accross N samples
            std_loss.append(get_std_loss(g, threshold=std_threshold[l]))
            if args.progressive_epoch is not None and l == epoch // args.progressive_epoch:
                # only calculate the var loss for active layers
                break
        std_loss = torch.stack(std_loss).sum()
        loss += args.std_loss_w * std_loss
        # cov_loss = torch.stack(cov_loss).mean()

    if args.cov_loss_w > 0:
        cov_loss = []
        for l, g in enumerate(gates):
            cov_loss.append(get_cov_loss(g))
            if args.progressive_epoch is not None and l == epoch // args.progressive_epoch:
                # only calculate the cov loss for active layers
                break
        cov_loss = torch.stack(cov_loss).mean()
        loss += args.cov_loss_w * cov_loss

    if args.dataset == "celeba":
        # need to scale mse
        out_psnr = out[mask_psnr].detach() if mask_psnr is not None else out.detach()
        y_psnr = y[mask_psnr].detach() if mask_psnr is not None else y.detach()
        mse = criterion((out_psnr + 1) / 2, (y_psnr + 1) / 2)
    elif args.dataset == 'srn':
        mse = criterion((out.detach() + 1) / 2, (y.detach() + 1) / 2)
    else:
        mse = mse.detach()
    psnr = 10 * np.log10(1 / mse.item())

    return loss, psnr


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def get_cov_loss(x):
    """
    Encourage the off-diagonal elements of the covariance matrix to be small"""
    N, C = x.shape
    x_norm = F.normalize(x, p=2, dim=1)
    cov_x = (x_norm.T @ x_norm) / (N - 1)
    return off_diagonal(cov_x).pow_(2).sum().div(C)


def get_std_loss(x, eps=1e-6, threshold=1e-1):
    """
    Encourage the std of the gate to be larger than a threshold
    """
    # first apply normalization to each row
    x_norm = F.normalize(x, p=2, dim=1)
    std_x = torch.sqrt(x_norm.var(dim=0) + eps)
    return torch.mean(F.relu(threshold - std_x))


def get_cv_loss(x, eps=1e-10):
    """The squared coefficient of variation of a sample.
    Useful as a loss to encourage a positive distribution to be more uniform.
    Epsilons added for numerical stability.
    Returns 0 for an empty Tensor.
    Args:
    x: a `Tensor`.
    Returns:
    a `Scalar`.
    """
    # if only num_experts = 1
    if x.shape[0] == 1:
        return torch.Tensor([0]).to(x.device)
    return x.float().var() / (x.float().mean() ** 2 + eps)


def entropy_regularization(output, beta=0.1):
    """
    Entropy regularization for the output
    """
    entropy = -torch.sum(output * torch.log(output + 1e-5), dim=1).mean()
    return beta * entropy


def interpolate(latent1, latent2, num_steps, condition_layer=None):
    """
    Interpolate between 2 latents
    """
    if condition_layer is None:
        return [
            (1 - alpha) * latent1 + alpha * latent2
            for alpha in torch.linspace(0, 1, num_steps)
        ]
    else:
        # latent shape: [layers, latent_size]
        latents = []
        for alpha in torch.linspace(0, 1, num_steps):
            latent = latent1.clone()
            latent[condition_layer] = (1 - alpha) * latent1[
                condition_layer
            ] + alpha * latent2[condition_layer]
            latents.append(latent)
        # latents: num_steps x [layers, latent_size]
        return latents


def render_interp_condition(args, epoch, model, latents, blend_alphas, num_steps=10):
    """
    Interpolate between 2 latents and render the images
    Determine the layer to interpolate, keep the other layer fixed
    """
    num_layer = latents.shape[1]
    grid_latents = []
    for l in range(num_layer):
        grid_latents.extend(
            interpolate(latents[0], latents[1], num_steps, condition_layer=l)
        )
        grid_latents.append(latents[1])
    grid_latents = torch.stack(
        grid_latents
    )  # [layers * (num_steps+1), layers, latent_size]

    model.eval()
    top_k = args.top_k and epoch >= args.warmup_epochs
    with torch.no_grad():
        coords = get_mgrid(args.side_length).cuda()
        out, _, _, _ = model(
            grid_latents, coords, top_k, blend_alphas=blend_alphas
        )
    out = out.reshape(-1, args.side_length, args.side_length, 3)
    out = out.permute(0, 3, 1, 2)
    out = torch.clamp(out, 0, 1)
    grid_samples = torchvision.utils.make_grid(out, nrow=num_steps + 1)
    if not os.path.exists(os.path.join(args.save, "interp_condition")):
        os.makedirs(os.path.join(args.save, "interp_condition"))
    torchvision.utils.save_image(
        grid_samples,
        os.path.join(
            args.save,
            "interp_condition",
            "interp_condition_train_e_{}.png".format(epoch),
        ),
    )


def render_interp(args, epoch, model, latents, blend_alphas, num_steps=10):
    """
    Interpolate between 4 latents and render the images
    """
    top_row = interpolate(latents[0], latents[1], num_steps)
    bottom_row = interpolate(latents[2], latents[3], num_steps)

    # Vertical interpolations
    grid_latents = []
    for top, bottom in zip(top_row, bottom_row):
        grid_latents.extend(interpolate(top, bottom, num_steps))
    grid_latents = torch.stack(grid_latents)  # [N, latent_size]

    # render the images
    model.eval()
    top_k = args.top_k and epoch >= args.warmup_epochs
    with torch.no_grad():
        coords = get_mgrid(args.side_length).cuda()
        out, _, _, _ = model(
            grid_latents, coords, top_k, blend_alphas=blend_alphas
        )  # N_imgs x N_coords x out_dim

    out = out.reshape(-1, args.side_length, args.side_length, 3)
    out = out.permute(0, 3, 1, 2)
    out = torch.clamp(out, 0, 1)
    grid_samples = torchvision.utils.make_grid(out, nrow=num_steps)
    if not os.path.exists(os.path.join(args.save, "interp")):
        os.makedirs(os.path.join(args.save, "interp"))
    torchvision.utils.save_image(
        grid_samples,
        os.path.join(args.save, "interp", "interp_train_e_{}.png".format(epoch)),
    )


def render_sample(args, epoch, model, blend_alphas):
    """
    Sample from the model and render the images
    """
    model.eval()
    top_k = args.top_k and epoch >= args.warmup_epochs
    with torch.no_grad():
        coords = get_mgrid(args.side_length).cuda()
        latents = torch.randn(16, args.latent_size).cuda() * args.std_latent
        out, _, _, _ = model(
            latents, coords, top_k, blend_alphas=blend_alphas
        )  # N_imgs x N_coords x out_dim
    out = out.reshape(-1, args.side_length, args.side_length, 3)
    out = out.permute(0, 3, 1, 2)
    out = torch.clamp(out, 0, 1)
    grid_samples = torchvision.utils.make_grid(out, nrow=4)
    if not os.path.exists(os.path.join(args.save, "sample")):
        os.makedirs(os.path.join(args.save, "sample"))
    torchvision.utils.save_image(
        grid_samples,
        os.path.join(args.save, "sample", "sample_train_e_{}.png".format(epoch)),
    )


def render(args, epoch, model, render_loader, blend_alphas, criterion, test=False):
    """
    Render the images, gates, and optionally the means(for conditional gate) and interpolations
    """
    model.eval()
    in_dict, gt_dict = next(iter(render_loader))
    if args.dataset == 'srn':
        in_dict_eval, gt_dict_eval = get_samples_for_nerf(
            args, copy.deepcopy(in_dict), copy.deepcopy(gt_dict),
            view_num=args.num_view_eval, pixel_sampling=False
        )
        img_eval = gt_dict_eval['img'].cuda()
        coords_eval = in_dict_eval['coords'].cuda()

        in_dict, gt_dict = get_samples_for_nerf(
            args, in_dict, gt_dict
        )
    img = gt_dict["img"].cuda()
    coords = in_dict["coords"].cuda()
    N = img.size(0)

    # reset the latents: random sample from N(0, 1)
    if args.gate_type in ["conditional", "separate"]:
        latents = torch.zeros(img.size(0), len(args.num_exps), args.latent_size).cuda()
    elif args.gate_type == "shared":
        latents = torch.zeros(img.size(0), args.latent_size).cuda()
    elif args.gate_type == 'direct':
        # initialize at 1/latent_size
        latents = torch.ones(img.size(0), len(args.num_exps), args.latent_size).cuda() / args.latent_size
    elif args.gate_type == 'hybrid':
        latents = torch.zeros(
            img.size(0), len(args.num_exps) + 1, args.latent_size
        ).cuda()
    else:
        raise ValueError("Invalid gate type")
    latents.requires_grad = True
    lr_inner_render = args.lr_inner * N / args.batch_size

    # prepare masks when using patchify and padding
    if args.side_patch > 1 and args.padding_ratio > 0:
        mask_all, mask_psnr, _ = get_masks(
            args.side_length, args.side_patch, args.padding_ratio
        )
    else:
        mask_all, mask_psnr = None, None

    if args.dataset == "celeba":
        C = img.size(1)
        if args.side_patch > 1 and args.padding_ratio > 0:
            img = patchify_with_padding(
                img, args.side_patch, args.padding_ratio
            )
        y = img.reshape(N, C, -1)
        y = y.permute(0, 2, 1)  # N_imgs x N_coords x 3
    elif args.dataset == "shapenet":
        y = img # N_imgs x N_coords x 3
    elif args.dataset == 'srn':
        y = img # N_imgs x N_coords x 3

    # meta_sgd
    if args.use_meta_sgd:
        meta_sgd_inner = model.meta_sgd_lrs()

    # determine if to use top_k
    top_k = args.top_k and epoch >= args.warmup_epochs

    # Inner loop: latents update
    for _ in range(args.inner_steps):
        out, gates, importance, _ = model(
            latents, coords, top_k, blend_alphas=blend_alphas
        )  # N_imgs x N_coords x out_dim
        if args.dataset == 'srn':
            out = nerf_volume_rendering(args, out, in_dict)
        loss, _ = compute_loss(args, epoch, out, y, criterion, gates, importance, top_k,
                               mask_all=mask_all, mask_psnr=mask_psnr)
        latent_gradients = torch.autograd.grad(loss, latents)[0]

        if args.use_meta_sgd:
            latents = latents - lr_inner_render * (meta_sgd_inner * latent_gradients)
        else:
            latents = latents - lr_inner_render * latent_gradients

    with torch.no_grad():
        if args.dataset == 'srn':
            out, gates, _, means = model(
                latents, coords_eval, top_k, blend_alphas=blend_alphas
            )
            out = nerf_volume_rendering(args, out, in_dict_eval, 'all')
        else: 
            out, gates, _, means = model(
                latents, coords, top_k, blend_alphas=blend_alphas
            )
        if mask_psnr is not None:
            # mask_psnr shape: H*W. Need to reshape to (N, H*W, C) as y
            mask_psnr = mask_psnr.unsqueeze(0).unsqueeze(-1).expand_as(out)
            out = out[mask_psnr]

    mode = "test" if test else "train"
    save_path = os.path.join(args.save, mode)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if args.dataset == "celeba":
        out = out.reshape(N, args.side_length, args.side_length, -1)
        out = out.permute(0, 3, 1, 2)
        # from -1, 1 to 0, 1
        out = (out + 1) / 2
        out = torch.clamp(out, 0, 1)
        grid_samples = torchvision.utils.make_grid(out, nrow=int(math.sqrt(N)))
        torchvision.utils.save_image(
            grid_samples, os.path.join(save_path, f"output_{mode}_e_{epoch}.png")
        )

    elif args.dataset == "shapenet":
        # find the valid coords (out >= 0.5)
        valid = (out >= 0.5).float().squeeze()
        valid_coords = [coords[i][valid[i].bool()].cpu().numpy() for i in range(N)]
        # plot the 3D scatter plot
        fig = plt.figure()
        M = int(math.sqrt(N))
        for i, points in enumerate(valid_coords):
            ax = fig.add_subplot(M, M, i + 1, projection="3d")
            x, y, z = points[:, 0], points[:, 1], points[:, 2]
            ax.scatter(x, y, z, s=0.5, c=z, cmap="rainbow")
            ax.view_init(elev=30, azim=45)
            # if points are all zeros, set the limit to be -1, 1
            if np.allclose(points, 0):
                min_lim, max_lim = -1, 1
            else:
                min_lim, max_lim = points.min(), points.max()
            ax.set_xlim(min_lim, max_lim)
            ax.set_ylim(min_lim, max_lim)
            ax.set_zlim(min_lim, max_lim)
            ax.set_box_aspect([1, 1, 1])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([min_lim, max_lim])
            ax.grid(False)
        plt.tight_layout()
        plt.savefig(
            os.path.join(save_path, f"output_{mode}_e_{epoch}.png"),
            bbox_inches="tight",
            dpi=300,
        )

    elif args.dataset == 'srn':
        save_rendering_output(out, img_eval, os.path.join(
            save_path, f'output_{mode}_e_{epoch}.png'
        ))

    if test:
        return

    # Uncomment to render the interpolation images
    # TODO: update the interp render for voxel
    # render_interp(args, epoch, model, latents[:4], blend_alphas=blend_alphas, num_steps=10)
    # render_interp_condition(args, epoch, model, latents[:2], blend_alphas=blend_alphas, num_steps=10)
    # render_sample(args, epoch, model, blend_alphas)

    # save the gates
    save_path = os.path.join(args.save, "gates")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    gates = [gate.detach().cpu().numpy() for gate in gates]
    fig, axes = plt.subplots(1, len(gates), figsize=(16, 6))
    for j, gate in enumerate(gates):
        cax = axes[j].imshow(gate, aspect="auto")
        axes[j].set_yticks([])
        axes[j].set_xticks([0, gate.shape[1] - 1])
        cbar = fig.colorbar(cax, ax=axes[j], orientation="horizontal")
        min_val, max_val = gate.min(), gate.max()
        cbar.set_ticks([min_val, max_val])
        # write title to be the mean std
        std_gate = gate.std(axis=0).mean()
        # normalize the gate and compute std
        gate_norm = gate / np.linalg.norm(gate, axis=1, keepdims=True)
        # gate_norm = gate - gate.mean(axis=0)
        std_gate_norm = gate_norm.std(axis=0).mean()
        axes[j].set_title("std: {:.4f} | std_norm: {:.4f}".format(std_gate, std_gate_norm))
    plt.subplots_adjust(wspace=0.4)
    plt.tight_layout()
    plt.savefig(
        os.path.join(save_path, f"gates_train_e_{epoch}.png"),
        bbox_inches="tight",
        dpi=300,
    )

    # save the means if available
    if means is not None:
        save_path = os.path.join(args.save, "means")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        means = [mean.detach().cpu().numpy() for mean in means]
        fig, axes = plt.subplots(1, len(means), figsize=(8, 2))
        for j, mean in enumerate(means):
            cax = axes[j].imshow(mean, aspect="auto")
            axes[j].set_yticks([])
            axes[j].set_xticks([0, mean.shape[1] - 1])
            cbar = fig.colorbar(cax, ax=axes[j], orientation="horizontal")
            min_val, max_val = mean.min(), mean.max()
            cbar.set_ticks([min_val, max_val])
            std_mean = mean.std(axis=0).mean()
            axes[j].set_title("std: {:.4f}".format(std_mean)) 
        plt.subplots_adjust(wspace=0.4)
        plt.tight_layout()
        plt.savefig(
            os.path.join(save_path, f"means_train_e_{epoch}.png"),
            bbox_inches="tight",
            dpi=300,
        )
    # close the figure
    plt.close("all")


def render_autodecoding(args, epoch, latents_all, model, render_loader, blend_alphas, criterion, test=False):
    """
    Render the images, gates, and optionally the means(for conditional gate) and interpolations
    """
    model.eval()
    in_dict, gt_dict = next(iter(render_loader))
    if args.dataset == 'srn':
        in_dict_eval, gt_dict_eval = get_samples_for_nerf(
            args, copy.deepcopy(in_dict), copy.deepcopy(gt_dict),
            view_num=args.num_view_eval, pixel_sampling=False
        )
        img_eval = gt_dict_eval['img'].cuda()
        coords_eval = in_dict_eval['coords'].cuda()

        in_dict, gt_dict = get_samples_for_nerf(
            args, in_dict, gt_dict
        )
    img = gt_dict["img"].cuda()
    idx = in_dict['idx'].cuda()
    coords = in_dict["coords"].cuda()
    N = img.size(0)

    latents = latents_all[idx].clone().detach()

    # prepare masks when using patchify and padding
    if args.side_patch > 1 and args.padding_ratio > 0:
        mask_all, mask_psnr, _ = get_masks(
            args.side_length, args.side_patch, args.padding_ratio
        )
    else:
        mask_all, mask_psnr = None, None

    if args.dataset == "celeba":
        C = img.size(1)
        if args.side_patch > 1 and args.padding_ratio > 0:
            img = patchify_with_padding(
                img, args.side_patch, args.padding_ratio
            )
        y = img.reshape(N, C, -1)
        y = y.permute(0, 2, 1)  # N_imgs x N_coords x 3
    elif args.dataset == "shapenet":
        y = img # N_imgs x N_coords x 3
    elif args.dataset == 'srn':
        y = img # N_imgs x N_coords x 3

    # determine if to use top_k
    top_k = args.top_k and epoch >= args.warmup_epochs

    with torch.no_grad():
        if args.dataset == 'srn':
            out, gates, _, means = model(
                latents, coords_eval, top_k, blend_alphas=blend_alphas
            )
            out = nerf_volume_rendering(args, out, in_dict_eval, 'all')
        else: 
            out, gates, _, means = model(
                latents, coords, top_k, blend_alphas=blend_alphas
            )
        if mask_psnr is not None:
            # mask_psnr shape: H*W. Need to reshape to (N, H*W, C) as y
            mask_psnr = mask_psnr.unsqueeze(0).unsqueeze(-1).expand_as(out)
            out = out[mask_psnr]

    mode = "test" if test else "train"
    save_path = os.path.join(args.save, mode)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if args.dataset == "celeba":
        out = out.reshape(N, args.side_length, args.side_length, -1)
        out = out.permute(0, 3, 1, 2)
        # from -1, 1 to 0, 1
        out = (out + 1) / 2
        out = torch.clamp(out, 0, 1)
        grid_samples = torchvision.utils.make_grid(out, nrow=int(math.sqrt(N)))
        torchvision.utils.save_image(
            grid_samples, os.path.join(save_path, f"output_{mode}_e_{epoch}.png")
        )

    elif args.dataset == "shapenet":
        # find the valid coords (out >= 0.5)
        valid = (out >= 0.5).float().squeeze()
        valid_coords = [coords[i][valid[i].bool()].cpu().numpy() for i in range(N)]
        # plot the 3D scatter plot
        fig = plt.figure()
        M = int(math.sqrt(N))
        for i, points in enumerate(valid_coords):
            ax = fig.add_subplot(M, M, i + 1, projection="3d")
            x, y, z = points[:, 0], points[:, 1], points[:, 2]
            ax.scatter(x, y, z, s=0.5, c=z, cmap="rainbow")
            ax.view_init(elev=30, azim=45)
            # if points are all zeros, set the limit to be -1, 1
            if np.allclose(points, 0):
                min_lim, max_lim = -1, 1
            else:
                min_lim, max_lim = points.min(), points.max()
            ax.set_xlim(min_lim, max_lim)
            ax.set_ylim(min_lim, max_lim)
            ax.set_zlim(min_lim, max_lim)
            ax.set_box_aspect([1, 1, 1])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([min_lim, max_lim])
            ax.grid(False)
        plt.tight_layout()
        plt.savefig(
            os.path.join(save_path, f"output_{mode}_e_{epoch}.png"),
            bbox_inches="tight",
            dpi=300,
        )

    elif args.dataset == 'srn':
        save_rendering_output(out, img_eval, os.path.join(
            save_path, f'output_{mode}_e_{epoch}.png'
        ))

    if test:
        return

    # Uncomment to render the interpolation images
    # TODO: update the interp render for voxel
    # render_interp(args, epoch, model, latents[:4], blend_alphas=blend_alphas, num_steps=10)
    # render_interp_condition(args, epoch, model, latents[:2], blend_alphas=blend_alphas, num_steps=10)
    # render_sample(args, epoch, model, blend_alphas)

    # save the gates
    save_path = os.path.join(args.save, "gates")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    gates = [gate.detach().cpu().numpy() for gate in gates]
    fig, axes = plt.subplots(1, len(gates), figsize=(16, 6))
    for j, gate in enumerate(gates):
        cax = axes[j].imshow(gate, aspect="auto")
        axes[j].set_yticks([])
        axes[j].set_xticks([0, gate.shape[1] - 1])
        cbar = fig.colorbar(cax, ax=axes[j], orientation="horizontal")
        min_val, max_val = gate.min(), gate.max()
        cbar.set_ticks([min_val, max_val])
        # write title to be the mean std
        std_gate = gate.std(axis=0).mean()
        # normalize the gate and compute std
        gate_norm = gate / np.linalg.norm(gate, axis=1, keepdims=True)
        # gate_norm = gate - gate.mean(axis=0)
        std_gate_norm = gate_norm.std(axis=0).mean()
        axes[j].set_title("std: {:.4f} | std_norm: {:.4f}".format(std_gate, std_gate_norm))
    plt.subplots_adjust(wspace=0.4)
    plt.tight_layout()
    plt.savefig(
        os.path.join(save_path, f"gates_train_e_{epoch}.png"),
        bbox_inches="tight",
        dpi=300,
    )

    # save the means if available
    if means is not None:
        save_path = os.path.join(args.save, "means")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        means = [mean.detach().cpu().numpy() for mean in means]
        fig, axes = plt.subplots(1, len(means), figsize=(8, 2))
        for j, mean in enumerate(means):
            cax = axes[j].imshow(mean, aspect="auto")
            axes[j].set_yticks([])
            axes[j].set_xticks([0, mean.shape[1] - 1])
            cbar = fig.colorbar(cax, ax=axes[j], orientation="horizontal")
            min_val, max_val = mean.min(), mean.max()
            cbar.set_ticks([min_val, max_val])
            std_mean = mean.std(axis=0).mean()
            axes[j].set_title("std: {:.4f}".format(std_mean)) 
        plt.subplots_adjust(wspace=0.4)
        plt.tight_layout()
        plt.savefig(
            os.path.join(save_path, f"means_train_e_{epoch}.png"),
            bbox_inches="tight",
            dpi=300,
        )
    # close the figure
    plt.close("all")


def compute_latents(args, epoch, model, data_loader, blend_alphas, criterion, test=False):
    # run inner loop to compute the latents for the dataset
    model.eval()

    latents_all = []
    means_all = []
    ssim = 0
    psnr = 0
    acc = 0
    rec = 0
    # use tqdm to show the progress bar
    for _, (in_dict, gt_dict) in enumerate(tqdm.tqdm(data_loader)):
        img = gt_dict["img"].cuda()
        coords = in_dict["coords"].cuda()
        N = img.size(0)

        # reset the latents: random sample from N(0, 1)
        if args.gate_type in ["conditional", "separate"]:
            latents = torch.zeros(
                img.size(0), len(args.num_exps), args.latent_size
            ).cuda()
        elif args.gate_type == "shared":
            latents = torch.zeros(img.size(0), args.latent_size).cuda()
        elif args.gate_type == 'direct':
            # initialize at 1/latent_size
            latents = torch.ones(img.size(0), len(args.num_exps), args.latent_size).cuda() / args.latent_size
        elif args.gate_type == 'hybrid':
            latents = torch.zeros(
                img.size(0), len(args.num_exps) + 1, args.latent_size
            ).cuda()
        else:
            raise ValueError("Invalid gate type")
        latents.requires_grad = True
        lr_inner_render = args.lr_inner * N / args.batch_size

        # prepare masks when using patchify and padding
        if args.side_patch > 1 and args.padding_ratio > 0:
            mask_all, mask_psnr, _ = get_masks(
                args.side_length, args.side_patch, args.padding_ratio
            )
        else:
            mask_all, mask_psnr = None, None

        if args.dataset == "celeba":
            C = img.size(1)
            if args.side_patch > 1 and args.padding_ratio > 0:
                img = patchify_with_padding(
                    img, args.side_patch, args.padding_ratio
                )
            y = img.reshape(N, C, -1)
            y = y.permute(0, 2, 1)
        elif args.dataset == "shapenet":
            y = img

        # meta_sgd
        if args.use_meta_sgd:
            meta_sgd_inner = model.meta_sgd_lrs()

        # determine if to use top_k
        top_k = args.top_k and epoch >= args.warmup_epochs

        # inner loop for latents update
        for _ in range(args.inner_steps):
            out, gates, importance, _ = model(latents, coords, top_k,
                                            blend_alphas=blend_alphas)
            loss, _ = compute_loss(args, epoch, out, y, criterion, gates, importance, top_k,
                                   mask_all=mask_all, mask_psnr=mask_psnr)
            latent_gradients = \
                    torch.autograd.grad(loss, latents)[0]
            
            if args.use_meta_sgd:
                latents = latents - lr_inner_render * (meta_sgd_inner * latent_gradients)
            else:
                latents = latents - lr_inner_render * latent_gradients

        with torch.no_grad():
            out, gates, importance, means = model(latents, coords, top_k,
                                              blend_alphas=blend_alphas)
        _, psnr_iter = compute_loss(args, epoch, out, y, criterion, gates, importance, top_k,
                                    mask_all=mask_all, mask_psnr=mask_psnr)
        ssim_iter = compute_ssim(out, y)

        ssim += ssim_iter
        psnr += psnr_iter
        if args.dataset == 'shapenet':
            pred = out >= 0.5
            pred = pred[mask_psnr] if mask_psnr is not None else pred
            acc += pred.float().eq(y).float().mean()
            rec += recall_score(y.cpu().numpy().flatten(), pred.cpu().numpy().flatten())

        latents_all.append(latents.detach())
        if means is not None:
            means = torch.stack(means, dim=1)
            means_all.append(means.detach())

    latents_all = torch.cat(latents_all, dim=0)
    split = 'test' if test else 'train'
    torch.save(latents_all, os.path.join(args.save, f"latents_{split}_e_{epoch}.pt"))
    if means_all:
        means_all = torch.cat(means_all, dim=0)
        torch.save(means_all, os.path.join(args.save, f"means_{split}_e_{epoch}.pt"))
    logging.info("Average PSNR: {:.4f}, SSIM: {:.4f}, Acc: {:.4f}, Recall: {:.4f}".format(
        psnr / len(data_loader),
        ssim / len(data_loader),
        acc / len(data_loader),
        rec / len(data_loader)))


def compute_ssim(img1, img2):
    """
    Compute the SSIM between 2 images of shape N x HW x C
    """
    N, HW, C = img1.shape
    H, W = int(math.sqrt(HW)), int(math.sqrt(HW))
    img1 = img1.reshape(N, H, W, C).cpu().numpy()
    img2 = img2.reshape(N, H, W, C).cpu().numpy()
    ssim_val = 0
    for i in range(N):
        ssim_val += ssim(img1[i], img2[i], data_range=1, channel_axis=-1)
    return ssim_val / N


def evaluate(args, epoch, model, data_loader, blend_alphas, criterion):
    """
    Evaluate the model on the test set
    """
    model.eval()
    ssim = 0
    psnr = 0
    acc = 0
    rec = 0
    for _, (in_dict, gt_dict) in enumerate(tqdm.tqdm(data_loader)):
        if args.dataset == 'srn':
            # in_dict_eval, gt_dict_eval = get_samples_for_nerf(
            #     args, copy.deepcopy(in_dict), copy.deepcopy(gt_dict),
            #     view_num=args.num_view_eval, pixel_sampling=False
            # )
            # img_eval = gt_dict_eval['img'].cuda()
            # coords_eval = in_dict_eval['coords'].cuda()

            in_dict, gt_dict = get_samples_for_nerf(
                args, in_dict, gt_dict
            )
        img = gt_dict["img"].cuda()
        coords = in_dict["coords"].cuda()
        N = img.size(0)

        # reset the latents: random sample from N(0, 1)
        if args.gate_type in ["conditional", "separate"]:
            latents = torch.zeros(
                img.size(0), len(args.num_exps), args.latent_size
            ).cuda()
        elif args.gate_type == "shared":
            latents = torch.zeros(img.size(0), args.latent_size).cuda()
        elif args.gate_type == 'direct':
            # initialize at 1/latent_size
            latents = torch.ones(img.size(0), len(args.num_exps), args.latent_size).cuda() / args.latent_size
        elif args.gate_type == 'hybrid':
            latents = torch.zeros(
                img.size(0), len(args.num_exps) + 1, args.latent_size
            ).cuda()
        else:
            raise ValueError("Invalid gate type")
        latents.requires_grad = True
        lr_inner_eval = args.lr_inner * N / args.batch_size

        # prepare masks when using patchify and padding
        if args.side_patch > 1 and args.padding_ratio > 0:
            mask_all, mask_psnr, _ = get_masks(
                args.side_length, args.side_patch, args.padding_ratio
            )
        else:
            mask_all, mask_psnr = None, None

        if args.dataset == "celeba":
            C = img.size(1)
            if args.side_patch > 1 and args.padding_ratio > 0:
                img = patchify_with_padding(
                    img, args.side_patch, args.padding_ratio
                )
            y = img.reshape(N, C, -1)
            y = y.permute(0, 2, 1)
        elif args.dataset == "shapenet":
            y = img
        elif args.dataset == 'srn':
            y = img # N_imgs x N_coords x 3

        # meta_sgd
        if args.use_meta_sgd:
            meta_sgd_inner = model.meta_sgd_lrs()

        # determine if to use top_k
        top_k = args.top_k and epoch >= args.warmup_epochs

        # inner loop for latents update
        for _ in range(args.inner_steps):
            out, gates, importance, _ = model(latents, coords, top_k,
                                            blend_alphas=blend_alphas)
            if args.dataset == 'srn':
                out = nerf_volume_rendering(args, out, in_dict)
            loss, _ = compute_loss(args, epoch, out, y, criterion, gates, importance, top_k,
                                   mask_all=mask_all, mask_psnr=mask_psnr)
            latent_gradients = \
                    torch.autograd.grad(loss, latents)[0]
            
            if args.use_meta_sgd:
                latents = latents - lr_inner_eval * (meta_sgd_inner * latent_gradients)
            else:
                latents = latents - lr_inner_eval * latent_gradients

        with torch.no_grad():
            out, gates, importance, _ = model(latents, coords, top_k,
                                              blend_alphas=blend_alphas)
            if args.dataset == 'srn':
                out = nerf_volume_rendering(args, out, in_dict)
        _, psnr_iter = compute_loss(args, epoch, out, y, criterion, gates, importance, top_k,
                                    mask_all=mask_all, mask_psnr=mask_psnr)
        
        ssim_iter = compute_ssim((out+1)/2, (y+1)/2)

        ssim += ssim_iter
        psnr += psnr_iter
        if args.dataset == 'shapenet':
            pred = out >= 0.5
            pred = pred[mask_psnr] if mask_psnr is not None else pred
            acc += pred.float().eq(y).float().mean()
            rec += recall_score(y.cpu().numpy().flatten(), pred.cpu().numpy().flatten())

    logging.info("Average Test PSNR: {:.4f}, SSIM: {:.4f}, Acc: {:.4f}, Recall: {:.4f}".format(
        psnr / len(data_loader),
        ssim / len(data_loader),
        acc / len(data_loader),
        rec / len(data_loader)))