import os
import sys
import time
import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from datasets import get_mgrid, LatticeDataset, CelebADataset
from model import INRLoe

import wandb
import argparse
import copy
import tqdm


def compute_loss(out, y, criterion, gates=None, importance=None, top_k=False, 
                 cv_loss_w=0, std_loss_w=0, cov_loss_w=0):
    '''
    Compute the loss for the model
    '''
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

    # Render the images
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


def render(args, epoch, model, render_loader, blend_alphas, test=False):
    '''
    Render the images, gates, and optionally the mu_var(for conditional gate) and interpolations
    '''
    model.eval()
    coords = get_mgrid(args.side_length).cuda()
    for i, (img, idx) in enumerate(render_loader):
        img, idx = img.cuda(), idx.cuda()
        N, C, H, W = img.shape
        out_all = []

        # Zero-initialize the latents
        if args.gate_type in ['conditional', 'separate']:
            latents = torch.zeros(img.size(0), len(args.num_exps), args.latent_size).cuda() 
        elif args.gate_type == 'shared':
            latents = torch.zeros(img.size(0), args.latent_size).cuda()
        else:
            raise ValueError("Invalid gate type")
        latents.requires_grad = True
        lr_inner_render = args.lr_inner * N / args.batch_size

        y = img.reshape(N, C, -1)
        y = y.permute(0, 2, 1) # N_imgs x N_coords x 3

        # Inner loop: latents update
        for _ in range(args.inner_steps):
            out, gates, importance, _ = model(latents, coords, args.top_k,
                                            blend_alphas=blend_alphas) # N_imgs x N_coords x out_dim
            loss, _ = compute_loss(out, y, criterion, gates, importance, 
                                    args.top_k, args.cv_loss, args.std_loss)
            latent_gradients = \
                    torch.autograd.grad(loss, latents)[0]

            latents = latents - lr_inner_render * latent_gradients
        
        with torch.no_grad():
            out, gates, _, mu_var = model(latents, coords, args.top_k,
                                    blend_alphas=blend_alphas)
        out_all.append(out)
        out_all = torch.cat(out_all, 1)
        out = out_all.reshape(N, args.side_length, args.side_length, C)
        out = out.permute(0, 3, 1, 2)
        out = torch.clamp(out, 0, 1)
        grid_samples = torchvision.utils.make_grid(out, nrow=int(math.sqrt(out.size(0))))

        if test:
            # Save the test images
            if not os.path.exists(os.path.join(args.save, "test")):
                os.makedirs(os.path.join(args.save, "test"))
            torchvision.utils.save_image(grid_samples, 
                                         os.path.join(args.save, "test", "output_test_e_{}.png".format(epoch)))
        else:
            # Save the train images
            if not os.path.exists(os.path.join(args.save, "train")):
                os.makedirs(os.path.join(args.save, "train"))
            torchvision.utils.save_image(grid_samples, 
                                         os.path.join(args.save, "train", "output_train_e_{}.png".format(epoch)))

            # Uncomment to render the interpolation images
            # render_interp(args, epoch, model, latents[:4], blend_alphas=blend_alphas, num_steps=10)
            # render_interp_condition(args, epoch, model, latents[:2], blend_alphas=blend_alphas, num_steps=10)
            # render_sample(args, epoch, model, blend_alphas)

            # Render the gates
            if not os.path.exists(os.path.join(args.save, "gates")):
                os.makedirs(os.path.join(args.save, "gates"))
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
            plt.savefig(os.path.join(args.save, "gates", "gates_train_e_{}.png".format(epoch)), 
                        bbox_inches='tight', dpi=300)
            
            # Render the mu_var for conditional gate
            if mu_var is not None:
                if not os.path.exists(os.path.join(args.save, "mu_var")):
                    os.makedirs(os.path.join(args.save, "mu_var"))
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
                plt.savefig(os.path.join(args.save, "mu_var", "mu_var_train_e_{}.png".format(epoch)), 
                            bbox_inches='tight', dpi=300)
        # close the figure
        plt.close('all')

def compute_latents(args, epoch, model, data_loader, blend_alphas):
    '''
    Run inner loops to compute the latents for the dataset
    '''
    model.eval()
    coords = get_mgrid(args.side_length).cuda()

    latents_all = []
    psnr = 0
    # use tqdm to show the progress bar
    for i, (img, _) in enumerate(tqdm.tqdm(data_loader)):
        img = img.cuda()
        N, C, _, _ = img.shape

        # Zero-initialize the latents
        if args.gate_type in ['conditional', 'separate']:
            latents = torch.zeros(img.size(0), len(args.num_exps), args.latent_size).cuda() 
        elif args.gate_type == 'shared':
            latents = torch.zeros(img.size(0), args.latent_size).cuda()
        else:
            raise ValueError("Invalid gate type")
        latents.requires_grad = True
        lr_inner_render = args.lr_inner * N / args.batch_size

        y = img.reshape(N, C, -1)
        y = y.permute(0, 2, 1)

        # Inner loop: latents update
        for _ in range(args.inner_steps):
            out, gates, importance, _ = model(latents, coords, args.top_k,
                                            blend_alphas=blend_alphas)
            loss, _ = compute_loss(out, y, criterion, gates, importance,
                                   args.top_k, args.cv_loss, args.std_loss)
            latent_gradients = \
                    torch.autograd.grad(loss, latents)[0]
            
            latents = latents - lr_inner_render * latent_gradients

        with torch.no_grad():
            out, gates, importance, _ = model(latents, coords, args.top_k,
                                              blend_alphas=blend_alphas)
        _, mse = compute_loss(out, y, criterion, gates, importance, 
                              args.top_k, args.cv_loss, args.std_loss)
        psnr += 10 * np.log10(1 / mse.item())

        latents_all.append(latents)
    latents_all = torch.cat(latents_all, 0)
    torch.save(latents_all, os.path.join(args.save, "latents_train_e_{}.pt".format(epoch)))
    logging.info("Average PSNR: {:.4f}".format(psnr / len(data_loader)))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='data/celeba64')
    parser.add_argument('--ckpt', type=str, default=None)

    # data loader
    parser.add_argument('--train_subset', type=int, default=30000)
    parser.add_argument('--render_subset', type=int, default=9)
    parser.add_argument('--side_length', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=14)

    # train params
    parser.add_argument('--epochs', type=int, default=600)
    parser.add_argument('--epochs_render', type=int, default=5)
    parser.add_argument('--epochs_save', type=int, default=5)
    parser.add_argument('--steps_log', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--min_lr', type=float, default=0, help='min learning rate')
    parser.add_argument('--lr_inner', type=float, default=1, help='learning rate for inner loop')
    parser.add_argument('--cv_loss', type=float, default=0, help='weight for cv loss')
    parser.add_argument('--std_loss', type=float, default=0, help='weight for std loss')
    parser.add_argument('--cov_loss', type=float, default=0, help='weight for cov loss')
    parser.add_argument('--inner_steps', type=int, default=3, help='number of inner steps for each coords')
    parser.add_argument('--grad_clip', type=float, default=1, help='gradient clipping')

    # model configs
    parser.add_argument('--top_k', action='store_true', help='whether to use top k sparce gates')
    parser.add_argument('--num_exps', nargs='+', type=int, default=[64, 64, 64, 64, 64])
    parser.add_argument('--ks', nargs='+', type=int, default=[8, 8, 8, 8, 8])
    parser.add_argument('--progressive_epoch', type=int, default=None, help='progressively enable experts for each layer')
    parser.add_argument('--progressive_reverse', action='store_true', help='reverse the progressive enablement')
    parser.add_argument('--latent_size', type=int, default=64, help='size of the latent for each layer')
    parser.add_argument('--num_hidden', type=int, default=4, help='number of hidden layers')
    parser.add_argument('--hidden_dim', type=int, default=64, help='hidden layer dim of each expert')
    parser.add_argument('--std_latent', type=float, default=0.0001, help='std of latent sampling')
    parser.add_argument('--gate_type', type=str, default='separate', help='gating type: separate, conditional, or shared')

    parser.add_argument('--compute_latents', action='store_true', help='compute latents for stage 2')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--save', type=str, default='save')
    parser.add_argument('--exp_cmt', type=str, default='DEFAULT')

    args = parser.parse_args()

    args.save = os.path.join(args.save, time.strftime('%Y%m%d_%H%M%S') + '_' + args.exp_cmt)
    os.makedirs(args.save, exist_ok=True)

    # Set up logging
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    # Log the arguments
    for arg in vars(args):
        if arg != 'root_dir':
            logging.info(f"{arg}: {getattr(args, arg)}")

    if args.wandb:
        wandb.init(project="inr-loe")
        wandb.config.update(args)

    inr_loe = INRLoe(hidden_dim=args.hidden_dim,
                     num_hidden=args.num_hidden,
                     num_exps=args.num_exps,
                     ks=args.ks,
                     latent_size=args.latent_size,
                     gate_type=args.gate_type,
                     ).cuda()

    # Count parameters
    params = sum(p.numel() for p in inr_loe.get_parameters())
    logging.info("Total number of parameters is: {}".format(params))
    logging.info("Model size is: {:.2f} MB".format(params * 4 / 1024**2)) 

    trainset = CelebADataset(root=args.root_dir, split='train', subset=args.train_subset, 
                            downsampled_size=(args.side_length, args.side_length))
    train_testset = CelebADataset(root=args.root_dir, split='train', subset=args.render_subset,
                            downsampled_size=(args.side_length, args.side_length))
    testset = CelebADataset(root=args.root_dir, split='test', subset=args.render_subset,
                            downsampled_size=(args.side_length, args.side_length))
    
    # When computing the latents, we don't need to shuffle the data
    dataloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=(not args.compute_latents), num_workers=4)
    train_testloader = torch.utils.data.DataLoader(train_testset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    optim_net = torch.optim.Adam(inr_loe.get_parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim_net, T_max=args.epochs, eta_min=args.min_lr)

    criterion = nn.MSELoss()

    if args.ckpt is not None:
        inr_loe.load_state_dict(torch.load(args.ckpt)) 
        start_epoch = int(args.ckpt.split('_')[-1].split('.')[0]) + 1
        logging.info("Model loaded from {}".format(args.ckpt))
        logging.info("Starting from epoch {}".format(start_epoch))
    else:
        start_epoch = 0

    # blend gating with uniform gating: alpha * uniform + (1 - alpha) * gates
    blend_alphas = [0] * len(inr_loe.num_exps)
    # blend_alphas[-1] = 1 # disable the last layer gating
    meta_grad_init = [0 for _ in inr_loe.get_parameters()]

    # Compute the latents for the dataset for stage 2
    if args.compute_latents:
        logging.info("Computing latents for the dataset...")
        compute_latents(args, start_epoch, inr_loe, dataloader, blend_alphas)
        # exit the program after computing the latents
        sys.exit()
    
    # Training epochs
    for epoch in range(start_epoch, args.epochs):
        inr_loe.train()
        coords = get_mgrid(args.side_length).cuda()

        # if progressive_epoch is not None, gradually enable the experts for each layer
        if args.progressive_epoch is not None:
            for l in range(1, len(blend_alphas)):
                # sigmoid function to gradually change the alpha
                alpha = 1 - 1 / (1 + math.exp((l * args.progressive_epoch - epoch) / 2))
                if args.progressive_reverse: 
                    blend_alphas[-l-1] = alpha
                else:
                    blend_alphas[l] = alpha

        psnr_epoch = 0
        # Outer loop: iterate over the images. 
        for i, (img, idx) in enumerate(dataloader):
            img = img.cuda() # N_imgs x 3 x 64 x 64
            idx = idx.cuda() # N_imgs

            if args.gate_type in ['conditional', 'separate']:
                latents = torch.randn(img.size(0), len(args.num_exps), args.latent_size).cuda() * args.std_latent
            elif args.gate_type == 'shared':
                latents = torch.randn(img.size(0), args.latent_size).cuda() * args.std_latent
            else:
                raise ValueError("Invalid gate type")
            latents.requires_grad = True

            # Initialise meta-gradient
            meta_grad = copy.deepcopy(meta_grad_init)

            N, C, H, W = img.shape
            y = img.reshape(N, C, -1)
            y = y.permute(0, 2, 1) # N_imgs x N_coords x 3

            # Inner loop: latents update
            for _ in range(args.inner_steps):
                out, gates, importance, _ = inr_loe(latents, coords, args.top_k, 
                                                blend_alphas=blend_alphas) # N_imgs x N_coords x out_dim
                loss, _ = compute_loss(out, y, criterion, gates, importance, 
                                        args.top_k, args.cv_loss, args.std_loss)
                latent_gradients = \
                    torch.autograd.grad(loss, latents, create_graph=True)[0]
                latents = latents - args.lr_inner * latent_gradients

            # Update the shared weights
            out, gates, importance, _ = inr_loe(latents, coords, args.top_k, 
                                            blend_alphas=blend_alphas)
            loss, mse = compute_loss(out, y, criterion, gates, importance, 
                                        args.top_k, args.cv_loss, args.std_loss)
            task_grad = torch.autograd.grad(loss, inr_loe.get_parameters())

            # Add to meta-gradient
            for g in range(len(task_grad)):
                meta_grad[g] += task_grad[g].detach()

            optim_net.zero_grad()
            for c, param in enumerate(inr_loe.get_parameters()):
                param.grad = meta_grad[c]

            nn.utils.clip_grad_norm_(inr_loe.get_parameters(), max_norm=args.grad_clip)
            optim_net.step()
            
            # Compute the PSNR
            mse = mse.item()
            psnr = 10 * np.log10(1 / mse) # shape is N_imgs

            psnr_epoch += psnr.mean()

            if i % args.steps_log == 0:
                logging.info("Epoch: {}, Iteration: {}, Loss: {:.4f}, PSNR: {:.4f}".format(
                    epoch, i, loss.item(), psnr.mean()))
                if args.wandb:
                    wandb.log({"loss": loss.item(), "psnr": psnr.mean()})
        scheduler.step()

        # Render the images
        if epoch % args.epochs_render == 0:
            render(args, epoch, inr_loe, train_testloader, blend_alphas)
            render(args, epoch, inr_loe, testloader, blend_alphas, test=True)

        # Loggings and saving the model
        psnr_epoch /= len(dataloader)
        logging.info("Epoch: {}, PSNR: {:.4f}".format(epoch, psnr_epoch))
        logging.info("Saving last model at epoch {}...".format(epoch))
        if not os.path.exists(os.path.join(args.save, "ckpt")):
            os.makedirs(os.path.join(args.save, "ckpt"))
        torch.save(inr_loe.state_dict(), os.path.join(args.save, "ckpt", "inr_loe_last.pt"))
        if epoch % args.epochs_save == 0:
            logging.info("Saving model...")
            torch.save(inr_loe.state_dict(), os.path.join(args.save, "ckpt", "inr_loe_{}.pt".format(epoch)))
        