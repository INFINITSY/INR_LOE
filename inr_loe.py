import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
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


def compute_loss(out, y, criterion, importance=None, top_k=False, cv_loss=0.01):
    mse = criterion(out, y)
    # if top_k: 
    loss = mse 
    # else:
    #     cv_loss = []
    #     for l, v in enumerate(importance):
    #         cv_loss.append(cv_squared_loss(v))
    #         if args.progressive_epoch is not None and l == epoch // args.progressive_epoch:
    #             # only calculate the cv loss for active layers
    #             break
    #     cv_loss = torch.stack(cv_loss).mean()
    #     loss = mse + cv_loss * cv_loss

    return loss, mse.detach()


def cv_squared_loss(x, eps=1e-10):
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
    # output: tensor of softmax outputs; shape: [N, C]
    # beta: regularization coefficient
    entropy = -torch.sum(output * torch.log(output + 1e-5), dim=1).mean()
    return beta * entropy

def interpolate(latent1, latent2, num_steps, condition_layer=None):
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
    # latents is a list of 2 tensors of shape [N, layers, latent_size]
    # first row, interp the first layer's latents, keep other layers the same
    num_layer = latents.shape[1]
    grid_latents = []
    for l in range(num_layer):
        grid_latents.extend(interpolate(latents[0], latents[1], num_steps, condition_layer=l)) 
        grid_latents.append(latents[1])
    grid_latents = torch.stack(grid_latents) # [layers * (num_steps+1), layers, latent_size]

    model.eval()
    with torch.no_grad():
        coords = get_mgrid(args.side_length).cuda()
        out, _, _, _ = model(grid_latents, coords, args.top_k, args.softmax,
                          blend_alphas=blend_alphas)
    out = out.reshape(-1, args.side_length, args.side_length, 3)
    out = out.permute(0, 3, 1, 2)
    out = torch.clamp(out, 0, 1)
    grid_samples = torchvision.utils.make_grid(out, nrow=num_steps+1)
    torchvision.utils.save_image(grid_samples, os.path.join(args.save, "interp_condition_train_e_{}.png".format(epoch)))


def render_interp(args, epoch, model, latents, blend_alphas, num_steps=10):
    # latents is a list of 4 latents
    # Horizontal interpolations (top and bottom)
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
        out, _, _, _ = model(grid_latents, coords, args.top_k, args.softmax,
                          blend_alphas=blend_alphas) # N_imgs x N_coords x out_dim

    out = out.reshape(-1, args.side_length, args.side_length, 3)
    out = out.permute(0, 3, 1, 2)
    out = torch.clamp(out, 0, 1)
    grid_samples = torchvision.utils.make_grid(out, nrow=num_steps)
    torchvision.utils.save_image(grid_samples, os.path.join(args.save, "interp_train_e_{}.png".format(epoch)))


def render_sample(args, epoch, model, blend_alphas):
    model.eval()
    with torch.no_grad():
        coords = get_mgrid(args.side_length).cuda()
        latents = torch.randn(16, args.latent_size).cuda() * args.std_latent
        out, _, _, _ = model(latents, coords, args.top_k, args.softmax,
                          blend_alphas=blend_alphas) # N_imgs x N_coords x out_dim
    out = out.reshape(-1, args.side_length, args.side_length, 3)
    out = out.permute(0, 3, 1, 2)
    out = torch.clamp(out, 0, 1)
    grid_samples = torchvision.utils.make_grid(out, nrow=4)
    torchvision.utils.save_image(grid_samples, os.path.join(args.save, "sample_train_e_{}.png".format(epoch)))


def render(args, epoch, model, render_loader, coords_loader, blend_alphas, test=False):
    model.eval()
    lr_inner_render = args.lr_inner * args.render_subset / args.batch_size
    
    for i, (img, idx) in enumerate(render_loader):
        img, idx = img.cuda(), idx.cuda()
        N, C, H, W = img.shape
        out_all = []

        # reset the latents: random sample from N(0, 1)
        if args.conditional:
            latents = torch.zeros(img.size(0), len(args.num_exps), args.latent_size).cuda() 
        else:
            latents = torch.zeros(img.size(0), args.latent_size).cuda()
        latents.requires_grad = True
        lr_inner_render = args.lr_inner * N / args.batch_size

        for coords, i_sel in coords_loader:
            coords, i_sel = coords.cuda(), i_sel.cuda()
            y = img.reshape(N, C, -1)[:, :, i_sel]
            y = y.permute(0, 2, 1) # N_imgs x N_coords x 3

            # inner loop for latents update
            for _ in range(args.inner_steps):
                out, _, importance, _ = model(latents, coords, args.top_k, args.softmax,
                                             blend_alphas=blend_alphas) # N_imgs x N_coords x out_dim
                loss, _ = compute_loss(out, y, criterion, importance, args.top_k, args.cv_loss)
                latent_gradients = \
                        torch.autograd.grad(loss, latents)[0]

                latents = latents - lr_inner_render * latent_gradients
            
            with torch.no_grad():
                out, gates, _, mu_var = model(latents, coords, args.top_k, args.softmax,
                                        blend_alphas=blend_alphas)
            out_all.append(out)
        
        out_all = torch.cat(out_all, 1)

        if args.patch_size is not None:
            n_patches = args.side_length // args.patch_size
            out_all = out_all.reshape(N, n_patches, n_patches, args.patch_size, args.patch_size, C)
            out_all = out_all.permute(0, 1, 3, 2, 4, 5) # [N, N_patches, P, N_patches, P, C]

        out = out_all.reshape(N, args.side_length, args.side_length, C)
        out = out.permute(0, 3, 1, 2)
        # clip the output to [0, 1]
        out = torch.clamp(out, 0, 1)
        # img = img.permute(0, 2, 3, 1)
        grid_samples = torchvision.utils.make_grid(out, nrow=int(math.sqrt(out.size(0))))
        if test:
            # plt.savefig(os.path.join(args.save, "output_test_e_{}.png".format(epoch)))
            torchvision.utils.save_image(grid_samples, 
                                         os.path.join(args.save, "output_test_e_{}.png".format(epoch)))
        else:
            # plt.savefig(os.path.join(args.save, "output_train_e_{}.png".format(epoch)))
            torchvision.utils.save_image(grid_samples, 
                                         os.path.join(args.save, "output_train_e_{}.png".format(epoch)))

            render_interp(args, epoch, model, latents[:4], blend_alphas=blend_alphas, num_steps=10)
            render_interp_condition(args, epoch, model, latents[:2], blend_alphas=blend_alphas, num_steps=10)
            # render_sample(args, epoch, model, blend_alphas)

            gates = [gate.detach().cpu().numpy() for gate in gates]
            fig, axes = plt.subplots(1, len(gates), figsize=(16, 4))
            for j, gate in enumerate(gates):
                cax = axes[j].imshow(gate, aspect='auto')
                axes[j].set_yticks([])
                axes[j].set_xticks([0, gate.shape[1]-1])
                cbar = fig.colorbar(cax, ax=axes[j], orientation='horizontal')
                min_val, max_val = gate.min(), gate.max()
                cbar.set_ticks([min_val, max_val])
            plt.subplots_adjust(wspace=0.4)
            plt.tight_layout()
            plt.savefig(os.path.join(args.save, "gates_train_e_{}.png".format(epoch)), 
                        bbox_inches='tight', dpi=300)
            
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
            plt.savefig(os.path.join(args.save, "mu_var_train_e_{}.png".format(epoch)), 
                        bbox_inches='tight', dpi=300)
        # close the figure
        plt.close('all')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='/home/s99tang/Research/loe_nvae/data/celeba')
    parser.add_argument('--ckpt', type=str, default=None)

    # data loader
    parser.add_argument('--train_subset', type=int, default=5000)
    parser.add_argument('--render_subset', type=int, default=9)
    parser.add_argument('--side_length', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=14)
    parser.add_argument('--chunk_size', type=int, default=4096)
    parser.add_argument('--patch_size', type=int, default=None)

    # train params
    parser.add_argument('--epochs', type=int, default=600)
    parser.add_argument('--epochs_render', type=int, default=10)
    parser.add_argument('--epochs_save', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--min_lr', type=float, default=0, help='min learning rate')
    parser.add_argument('--lr_inner', type=float, default=1, help='learning rate for inner loop')
    parser.add_argument('--cv_loss', type=float, default=0, help='weight for cv loss')
    parser.add_argument('--inner_steps', type=int, default=5, help='number of inner steps for each coords')
    parser.add_argument('--grad_clip', type=float, default=1, help='gradient clipping')

    # model configs
    parser.add_argument('--top_k', action='store_true', help='whether to use top k sparce gates')
    parser.add_argument('--softmax', action='store_true', help='whether to use softmax on gates')
    parser.add_argument('--bias', action='store_true', help='use bias on weighted experts')
    parser.add_argument('--merge_before_act', type=bool, default=True, help='merge experts before nl act')
    parser.add_argument('--num_exps', nargs='+', type=int, default=[64, 64, 64, 64])
    parser.add_argument('--ks', nargs='+', type=int, default=[8, 8, 8, 8])
    parser.add_argument('--progressive_epoch', type=int, default=None, help='progressively enable experts for each layer')
    parser.add_argument('--progressive_reverse', action='store_true', help='reverse the progressive enablement')
    parser.add_argument('--latent_size', type=int, default=256, help='size of the latent space')
    parser.add_argument('--num_hidden', type=int, default=3, help='number of hidden layers')
    parser.add_argument('--hidden_dim', type=int, default=64, help='hidden layer dim')
    parser.add_argument('--std_latent', type=float, default=0.0001, help='std of latent sampling')
    parser.add_argument('--conditional', action='store_true', help='use layer-wise conditional latents')

    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--save', type=str, default='save')
    parser.add_argument('--exp_cmt', type=str, default='conditional_latent')

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

    # log all input arguments
    for arg in vars(args):
        if arg != 'root_dir':
            logging.info(f"{arg}: {getattr(args, arg)}")

    if args.wandb:
        wandb.init(project="inr-loe")
        wandb.config.update(args)

    coords = get_mgrid(args.side_length).cuda()
    inr_loe = INRLoe(hidden_dim=args.hidden_dim,
                     num_hidden=args.num_hidden,
                     num_exps=args.num_exps,
                     ks=args.ks,
                     image_resolution=args.side_length, 
                     merge_before_act=args.merge_before_act, 
                     bias=args.bias,
                     patchwise=(args.patch_size is not None),
                     latent_size=args.latent_size,
                     conditional=args.conditional
                     ).cuda()

    # count the number of parameters
    params = sum(p.numel() for p in inr_loe.parameters())
    logging.info("Total number of parameters is: {}".format(params))
    logging.info("Model size is: {:.2f} MB".format(params * 4 / 1024**2)) 

    # dataset = CIFAR10Dataset(root='data', class_label=5)
    trainset = CelebADataset(root=args.root_dir, split='train', subset=args.train_subset, 
                            downsampled_size=(args.side_length, args.side_length))
    train_testset = CelebADataset(root=args.root_dir, split='train', subset=args.render_subset,
                            downsampled_size=(args.side_length, args.side_length))
    testset = CelebADataset(root=args.root_dir, split='test', subset=args.render_subset,
                            downsampled_size=(args.side_length, args.side_length))
    
    dataloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    train_testloader = torch.utils.data.DataLoader(train_testset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    img_size = args.side_length if args.patch_size is None else args.patch_size
    coordset = LatticeDataset(image_shape=(img_size, img_size))
    coords_loader = torch.utils.data.DataLoader(coordset, pin_memory=True, batch_size=args.chunk_size, shuffle=True)
    valid_coords = torch.utils.data.DataLoader(coordset, pin_memory=True, batch_size=args.chunk_size, shuffle=False)

    optim_net = torch.optim.Adam(inr_loe.parameters(), lr=args.lr, weight_decay=1e-5)
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
    meta_grad_init = [0 for _ in inr_loe.parameters()]

    for epoch in range(start_epoch, args.epochs):
        inr_loe.train()

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
        # iterate over the images
        for i, (img, idx) in enumerate(dataloader):
            img = img.cuda() # N_imgs x 3 x 64 x 64
            idx = idx.cuda() # N_imgs
            # N_imgs = img.shape[0]
            if args.patch_size is not None:
                img_perm = img.permute(0, 2, 3, 1) # N_imgs x 64 x 64 x 3
                n_patches = args.side_length // args.patch_size
                img_reshaped = img_perm.reshape(img_perm.size(0), n_patches, args.patch_size, 
                                                n_patches, args.patch_size, img_perm.size(-1))
                img_transposed = img_reshaped.permute(0, 1, 3, 2, 4, 5)
                img_patch = img_transposed.reshape(-1, args.patch_size, args.patch_size, img_perm.size(-1))
                # img_patch: [N_imgs x N_patches x N_patches] x 16 x 16 x 3
                img_gt = img_patch.permute(0, 3, 1, 2)
            else:
                img_gt = img

            # reset the latents: random sample from N(0, 1) 
            # shape: N_imgs x latent_size
            if args.conditional:
                latents = torch.randn(img.size(0), len(args.num_exps), args.latent_size).cuda() * args.std_latent
            else:
                latents = torch.randn(img.size(0), args.latent_size).cuda() * args.std_latent
            latents.requires_grad = True

            # initialise meta-gradient
            meta_grad = copy.deepcopy(meta_grad_init)

            # iterate over the coords
            for coords, i_sel in coords_loader:
                coords, i_sel = coords.cuda(), i_sel.cuda()
                N, C, H, W = img_gt.shape
                y = img_gt.reshape(N, C, -1)[:, :, i_sel]
                y = y.permute(0, 2, 1) # N_imgs x N_coords x 3

                # inner loop for latents update
                for _ in range(args.inner_steps):
                    out, gates, importance, _ = inr_loe(latents, coords, args.top_k, args.softmax, 
                                                 blend_alphas=blend_alphas) # N_imgs x N_coords x out_dim

                    loss, _ = compute_loss(out, y, criterion, importance, args.top_k, args.cv_loss)
                    latent_gradients = \
                        torch.autograd.grad(loss, latents, create_graph=True)[0]
                    latents = latents - args.lr_inner * latent_gradients

                # update the shared weights
                out, gates, importance, _ = inr_loe(latents, coords, args.top_k, args.softmax, 
                                             blend_alphas=blend_alphas)
                
                loss, mse = compute_loss(out, y, criterion, importance, args.top_k, args.cv_loss)
                task_grad = torch.autograd.grad(loss, inr_loe.parameters())

                # add to meta-gradient
                for g in range(len(task_grad)):
                    meta_grad[g] += task_grad[g].detach()

                optim_net.zero_grad()
                for c, param in enumerate(inr_loe.parameters()):
                    param.grad = meta_grad[c]

                nn.utils.clip_grad_norm_(inr_loe.parameters(), max_norm=args.grad_clip)
                optim_net.step()
                
                # compute psnr
                mse = mse.item()
                psnr = 10 * np.log10(1 / mse) # shape is N_imgs

                psnr_epoch += psnr.mean()

            if i % 25 == 0:
                logging.info("Epoch: {}, Iteration: {}, Loss: {:.4f}, PSNR: {:.4f}".format(
                    epoch, i, loss.item(), psnr.mean()))
                if args.wandb:
                    wandb.log({"loss": loss.item(), "psnr": psnr.mean()})
        scheduler.step()

        if epoch % args.epochs_render == 0:
            render(args, epoch, inr_loe, train_testloader, valid_coords, blend_alphas)
            render(args, epoch, inr_loe, testloader, valid_coords, blend_alphas, test=True)

        psnr_epoch /= len(dataloader)
        logging.info("Epoch: {}, PSNR: {:.4f}".format(epoch, psnr_epoch))

        logging.info("Saving last model at epoch {}...".format(epoch))
        torch.save(inr_loe.state_dict(), os.path.join(args.save, "inr_loe_last.pt"))
        if epoch % args.epochs_save == 0:
            logging.info("Saving model...")
            torch.save(inr_loe.state_dict(), os.path.join(args.save, "inr_loe_{}.pt".format(epoch)))
        