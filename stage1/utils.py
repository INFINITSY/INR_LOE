import os
import math
import tqdm
import torch
import torchvision
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import logging

from sklearn.metrics import recall_score


def get_mgrid(sidelen, dim=2, max=1.0):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int'''
    tensors = tuple(dim * [torch.linspace(-max, max, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid


def compute_loss(args, epoch, out, y, criterion, gates=None, importance=None, top_k=False, 
                 cv_loss_w=0, std_loss_w=0, cov_loss_w=0):
    mse = criterion(out, y)
    if top_k: 
        loss = mse 
    else:
        if importance is not None:
            cv_loss = []
            for l, v in enumerate(importance):
                cv_loss.append(get_cv_loss(v))
                if args.progressive_epoch is not None and l == epoch // args.progressive_epoch:
                    # only calculate the cv loss for active layers
                    break
            cv_loss = torch.stack(cv_loss).mean()
        else:
            cv_loss = 0
        
        if gates is not None:
            std_loss = []
            cov_loss = []
            # gates is a list: num_layers x [N, num_experts]
            # std_threshold = [0.005, 0.01, 0.05, 0.1, 0.15]
            std_threshold = [0.01] * len(gates)
            for l, g in enumerate(gates):
                # compute the variance accross N samples
                std_loss.append(get_std_loss(g, threshold=std_threshold[l]))
                cov_loss.append(get_cov_loss(g))
                if args.progressive_epoch is not None and l == epoch // args.progressive_epoch:
                    # only calculate the var loss for active layers
                    break
            std_loss = torch.stack(std_loss).sum()
            cov_loss = torch.stack(cov_loss).mean()
        else:
            std_loss = 0
            cov_loss = 0
        
        loss = mse + cv_loss_w * cv_loss + std_loss_w * std_loss + cov_loss_w * cov_loss

    return loss, mse.detach()

def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def get_cov_loss(x):
    '''
    Encourage the off-diagonal elements of the covariance matrix to be small'''
    N, C = x.shape
    cov_x = (x.T @ x) / (N - 1)
    return off_diagonal(cov_x).pow_(2).sum().div(C)

def get_std_loss(x, eps=1e-6, threshold=1e-1):
    '''
    Encourage the std of the gate to be larger than a threshold
    '''
    std_x = torch.sqrt(x.var(dim=0) + eps)
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
    return x.float().var() / (x.float().mean()**2 + eps)

def entropy_regularization(output, beta=0.1):
    '''
    Entropy regularization for the output
    '''
    entropy = -torch.sum(output * torch.log(output + 1e-5), dim=1).mean()
    return beta * entropy

def interpolate(latent1, latent2, num_steps, condition_layer=None):
    '''
    Interpolate between 2 latents
    '''
    if condition_layer is None:
        return [(1 - alpha) * latent1 + alpha * latent2 for alpha in torch.linspace(0, 1, num_steps)]
    else:
        # latent shape: [layers, latent_size]
        latents = []
        for alpha in torch.linspace(0, 1, num_steps):
            latent = latent1.clone()
            latent[condition_layer] = (1 - alpha) * latent1[condition_layer] + alpha * latent2[condition_layer]
            latents.append(latent)
        # latents: num_steps x [layers, latent_size]
        return latents

def render_interp_condition(args, epoch, model, latents, blend_alphas, num_steps=10):
    '''
    Interpolate between 2 latents and render the images
    Determine the layer to interpolate, keep the other layer fixed
    '''
    num_layer = latents.shape[1]
    grid_latents = []
    for l in range(num_layer):
        grid_latents.extend(interpolate(latents[0], latents[1], num_steps, condition_layer=l)) 
        grid_latents.append(latents[1])
    grid_latents = torch.stack(grid_latents) # [layers * (num_steps+1), layers, latent_size]

    model.eval()
    with torch.no_grad():
        coords = get_mgrid(args.side_length).cuda()
        out, _, _, _ = model(grid_latents, coords, args.top_k,
                          blend_alphas=blend_alphas)
    out = out.reshape(-1, args.side_length, args.side_length, 3)
    out = out.permute(0, 3, 1, 2)
    out = torch.clamp(out, 0, 1)
    grid_samples = torchvision.utils.make_grid(out, nrow=num_steps+1)
    if not os.path.exists(os.path.join(args.save, "interp_condition")):
        os.makedirs(os.path.join(args.save, "interp_condition"))
    torchvision.utils.save_image(grid_samples, os.path.join(args.save, "interp_condition", 
                                                            "interp_condition_train_e_{}.png".format(epoch)))


def render_interp(args, epoch, model, latents, blend_alphas, num_steps=10):
    '''
    Interpolate between 4 latents and render the images
    '''
    top_row = interpolate(latents[0], latents[1], num_steps)
    bottom_row = interpolate(latents[2], latents[3], num_steps)

    # Vertical interpolations
    grid_latents = []
    for top, bottom in zip(top_row, bottom_row):
        grid_latents.extend(interpolate(top, bottom, num_steps))
    grid_latents = torch.stack(grid_latents) # [N, latent_size]

    # render the images
    model.eval()
    with torch.no_grad():
        coords = get_mgrid(args.side_length).cuda()
        out, _, _, _ = model(grid_latents, coords, args.top_k,
                          blend_alphas=blend_alphas) # N_imgs x N_coords x out_dim

    out = out.reshape(-1, args.side_length, args.side_length, 3)
    out = out.permute(0, 3, 1, 2)
    out = torch.clamp(out, 0, 1)
    grid_samples = torchvision.utils.make_grid(out, nrow=num_steps)
    if not os.path.exists(os.path.join(args.save, "interp")):
        os.makedirs(os.path.join(args.save, "interp"))
    torchvision.utils.save_image(grid_samples, os.path.join(args.save, "interp", 
                                                            "interp_train_e_{}.png".format(epoch)))


def render_sample(args, epoch, model, blend_alphas):
    '''
    Sample from the model and render the images
    '''
    model.eval()
    with torch.no_grad():
        coords = get_mgrid(args.side_length).cuda()
        latents = torch.randn(16, args.latent_size).cuda() * args.std_latent
        out, _, _, _ = model(latents, coords, args.top_k,
                          blend_alphas=blend_alphas) # N_imgs x N_coords x out_dim
    out = out.reshape(-1, args.side_length, args.side_length, 3)
    out = out.permute(0, 3, 1, 2)
    out = torch.clamp(out, 0, 1)
    grid_samples = torchvision.utils.make_grid(out, nrow=4)
    if not os.path.exists(os.path.join(args.save, "sample")):
        os.makedirs(os.path.join(args.save, "sample"))
    torchvision.utils.save_image(grid_samples, os.path.join(args.save, "sample", 
                                                            "sample_train_e_{}.png".format(epoch)))


def render(args, epoch, model, render_loader, blend_alphas, criterion, test=False):
    '''
    Render the images, gates, and optionally the mu_var(for conditional gate) and interpolations
    '''
    model.eval()
    in_dict, gt_dict = next(iter(render_loader))
    img = gt_dict['img'].cuda()
    coords = in_dict['coords'].cuda()
    N = img.size(0)

    # reset the latents: random sample from N(0, 1)
    if args.gate_type in ['conditional', 'separate']:
        latents = torch.zeros(img.size(0), len(args.num_exps), args.latent_size).cuda() 
    elif args.gate_type == 'shared':
        latents = torch.zeros(img.size(0), args.latent_size).cuda()
    else:
        raise ValueError("Invalid gate type")
    latents.requires_grad = True
    lr_inner_render = args.lr_inner * N / args.batch_size

    if args.dataset == 'celeba':
        C = img.size(1)
        y = img.reshape(N, C, -1)
        y = y.permute(0, 2, 1) # N_imgs x N_coords x 3
    elif args.dataset == 'shapenet':
        y = img

    # Inner loop: latents update
    for _ in range(args.inner_steps):
        out, gates, importance, _ = model(latents, coords, args.top_k,
                                        blend_alphas=blend_alphas) # N_imgs x N_coords x out_dim
        loss, _ = compute_loss(args, epoch, out, y, criterion, gates, importance, 
                                args.top_k, args.cv_loss, args.std_loss)
        latent_gradients = \
                torch.autograd.grad(loss, latents)[0]

        latents = latents - lr_inner_render * latent_gradients
    
    with torch.no_grad():
        out, gates, _, mu_var = model(latents, coords, args.top_k,
                                blend_alphas=blend_alphas)
    
    mode = 'test' if test else 'train'
    save_path = os.path.join(args.save, mode)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if args.dataset == 'celeba':
        out = out.reshape(N, args.side_length, args.side_length, -1)
        out = out.permute(0, 3, 1, 2)
        out = torch.clamp(out, 0, 1)
        grid_samples = torchvision.utils.make_grid(out, nrow=int(math.sqrt(N)))
        torchvision.utils.save_image(grid_samples, 
                                     os.path.join(save_path, f"output_{mode}_e_{epoch}.png"))
    
    elif args.dataset == 'shapenet':
        # find the valid coords (out >= 0.5)
        valid = (out >= 0.5).float().squeeze()
        valid_coords = [coords[i][valid[i].bool()].cpu().numpy() for i in range(N)]
        # plot the 3D scatter plot
        fig = plt.figure()
        M = int(math.sqrt(N))
        for i, points in enumerate(valid_coords):
            ax = fig.add_subplot(M,M,i+1, projection='3d')
            x, y, z = points[:, 0], points[:, 1], points[:, 2]
            ax.scatter(x, y, z, s=0.5, c=z, cmap='rainbow')
            ax.view_init(elev=30, azim=45) 
            # if points are all zeros, set the limit to be -1, 1
            if np.allclose(points, 0):
                min_lim, max_lim = -1, 1
            else:
                min_lim, max_lim = points.min(), points.max()
            ax.set_xlim(min_lim, max_lim)
            ax.set_ylim(min_lim, max_lim)
            ax.set_zlim(min_lim, max_lim)
            ax.set_box_aspect([1,1,1])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([min_lim, max_lim])
            ax.grid(False)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"output_{mode}_e_{epoch}.png"), 
                    bbox_inches='tight', dpi=300)
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
        cax = axes[j].imshow(gate, aspect='auto')
        axes[j].set_yticks([])
        axes[j].set_xticks([0, gate.shape[1]-1])
        cbar = fig.colorbar(cax, ax=axes[j], orientation='horizontal')
        min_val, max_val = gate.min(), gate.max()
        cbar.set_ticks([min_val, max_val])
        # write title to be the mean std
        std_gate = gate.std(axis=0).mean()
        axes[j].set_title("std: {:.4f}".format(std_gate))
    plt.subplots_adjust(wspace=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"gates_train_e_{epoch}.png"), 
                bbox_inches='tight', dpi=300)
    
    # save the mu_var if available
    if mu_var is not None:
        save_path = os.path.join(args.save, "mu_var")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        means, log_vars = mu_var['means'], mu_var['log_vars']
        means = [mean.detach().cpu().numpy() for mean in means]
        log_vars = [log_var.detach().cpu().numpy() for log_var in log_vars]
        fig, axes = plt.subplots(2, len(means), figsize=(8, 4))
        for j, (mean, log_var) in enumerate(zip(means, log_vars)):
            cax = axes[0, j].imshow(mean, aspect='auto')
            axes[0, j].set_yticks([])
            axes[0, j].set_xticks([0, mean.shape[1]-1])
            cbar = fig.colorbar(cax, ax=axes[0, j], orientation='horizontal')
            min_val, max_val = mean.min(), mean.max()
            cbar.set_ticks([min_val, max_val])

            cax = axes[1, j].imshow(log_var, aspect='auto')
            axes[1, j].set_yticks([])
            axes[1, j].set_xticks([0, log_var.shape[1]-1])
            cbar = fig.colorbar(cax, ax=axes[1, j], orientation='horizontal')
            min_val, max_val = log_var.min(), log_var.max()
            cbar.set_ticks([min_val, max_val])
        plt.subplots_adjust(wspace=0.4)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"mu_var_train_e_{epoch}.png"), 
                    bbox_inches='tight', dpi=300)
    # close the figure
    plt.close('all')

def compute_latents(args, epoch, model, data_loader, blend_alphas, criterion, test=False):
    # run inner loop to compute the latents for the dataset
    model.eval()

    latents_all = []
    psnr = 0
    acc = 0
    rec = 0
    # use tqdm to show the progress bar
    for _, (in_dict, gt_dict) in enumerate(tqdm.tqdm(data_loader)):
        img = gt_dict['img'].cuda()
        coords = in_dict['coords'].cuda()
        N = img.size(0)

        # reset the latents: random sample from N(0, 1)
        if args.gate_type in ['conditional', 'separate']:
            latents = torch.zeros(img.size(0), len(args.num_exps), args.latent_size).cuda() 
        elif args.gate_type == 'shared':
            latents = torch.zeros(img.size(0), args.latent_size).cuda()
        else:
            raise ValueError("Invalid gate type")
        latents.requires_grad = True
        lr_inner_render = args.lr_inner * N / args.batch_size

        if args.dataset == 'celeba':
            C = img.size(1)
            y = img.reshape(N, C, -1)
            y = y.permute(0, 2, 1)
        elif args.dataset == 'shapenet':
            y = img

        # inner loop for latents update
        for _ in range(args.inner_steps):
            out, gates, importance, _ = model(latents, coords, args.top_k,
                                            blend_alphas=blend_alphas)
            loss, _ = compute_loss(args, epoch, out, y, criterion, gates, importance,
                                   args.top_k, args.cv_loss, args.std_loss)
            latent_gradients = \
                    torch.autograd.grad(loss, latents)[0]
            
            latents = latents - lr_inner_render * latent_gradients

        with torch.no_grad():
            out, gates, importance, _ = model(latents, coords, args.top_k,
                                              blend_alphas=blend_alphas)
        _, mse = compute_loss(args, epoch, out, y, criterion, gates, importance, 
                              args.top_k, args.cv_loss, args.std_loss)
        psnr += 10 * np.log10(1 / mse.item())
        if args.dataset == 'shapenet':
            pred = out >= 0.5
            acc += pred.float().eq(y).float().mean()
            rec += recall_score(y.cpu().numpy().flatten(), pred.cpu().numpy().flatten())

        latents_all.append(latents)
    latents_all = torch.cat(latents_all, 0)
    split = 'test' if test else 'train'
    torch.save(latents_all, os.path.join(args.save, f"latents_{split}_e_{epoch}.pt"))
    logging.info("Average PSNR: {:.4f}, Acc: {:.4f}, Recall: {:.4f}".format(psnr / len(data_loader),
                                                                            acc / len(data_loader),
                                                                            rec / len(data_loader)))