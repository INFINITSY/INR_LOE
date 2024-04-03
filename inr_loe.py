import os
import sys
import time
import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from datasets import get_mgrid, LatticeDataset, CelebADataset
from model import INRLoe

import wandb
import argparse

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

@torch.no_grad()
def render(args, epoch, model, render_loader, coords_loader, blend_alphas, test=False):
    model.eval()
    
    for i, (img, idx) in enumerate(render_loader):
        img, idx = img.cuda(), idx.cuda()
        N, C, H, W = img.shape
        out_all = []
        for coords, i_sel in coords_loader:
            coords, i_sel = coords.cuda(), i_sel.cuda()

            out, gates, _ = model(img, coords, args.top_k, args.softmax,
                                  blend_alphas=blend_alphas)
            
            out_all.append(out)
        
        out_all = torch.cat(out_all, 1)

        if args.patch_size is not None:
            n_patches = args.side_length // args.patch_size
            out_all = out_all.reshape(N, n_patches, n_patches, args.patch_size, args.patch_size, C)
            out_all = out_all.permute(0, 1, 3, 2, 4, 5) # [N, N_patches, P, N_patches, P, C]

        out = out_all.reshape(N, args.side_length, args.side_length, C)
        # clip the output to [0, 1]
        out = torch.clamp(out, 0, 1)
        img = img.permute(0, 2, 3, 1)
        # display img and out side by side
        fig, axes = plt.subplots(N, 2, figsize=(2, 5))
        for i in range(N):
            axes[i, 0].imshow(img[i].cpu())
            axes[i, 1].imshow(out[i].cpu())

        if test:
            plt.savefig(os.path.join(args.save, "output_test_e_{}.png".format(epoch)))
        else:
            plt.savefig(os.path.join(args.save, "output_train_e_{}.png".format(epoch)))

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
        # close the figure
        plt.close('all')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='/home/s99tang/Research/neural_dynamic/data/celeba64')
    parser.add_argument('--ckpt', type=str, default=None)

    # data loader
    parser.add_argument('--train_subset', type=int, default=50)
    parser.add_argument('--render_subset', type=int, default=3)
    parser.add_argument('--side_length', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=3)
    parser.add_argument('--chunk_size', type=int, default=1024)
    parser.add_argument('--patch_size', type=int, default=16)

    # train params
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--epochs_render', type=int, default=1)
    parser.add_argument('--epochs_save', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--cv_loss', type=float, default=0.01, help='weight for cv loss')
    parser.add_argument('--separate_optim', type=bool, default=True, help='separately update net and gates')

    # model configs
    parser.add_argument('--top_k', action='store_true', help='whether to use top k sparce gates')
    parser.add_argument('--softmax', type=bool, default=True, help='whether to use softmax on gates')
    parser.add_argument('--bias', action='store_true', help='use bias on weighted experts')
    parser.add_argument('--merge_before_act', type=bool, default=True, help='merge experts before nl act')
    parser.add_argument('--num_exps', nargs='+', type=int, default=[8, 16, 64, 256, 1024])
    parser.add_argument('--ks', nargs='+', type=int, default=[4, 8, 32, 128, 256])
    parser.add_argument('--progressive_epoch', type=int, default=10, help='progressively enable experts for each layer')
    parser.add_argument('--progressive_reverse', action='store_true', help='reverse the progressive enablement')
    
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--save', type=str, default='save')
    parser.add_argument('--exp_cmt', type=str, default='patch_test_res18')

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
    inr_loe = INRLoe(num_exps=args.num_exps,
                     ks=args.ks,
                     image_resolution=args.side_length, 
                     merge_before_act=args.merge_before_act, 
                     bias=args.bias).cuda()

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

    optim_gates = torch.optim.Adam(inr_loe.gate_module.parameters(), lr=args.lr, weight_decay=1e-5)
    optim_net = torch.optim.Adam(inr_loe.net.parameters(), lr=args.lr, weight_decay=1e-5)

    criterion = nn.MSELoss()

    if args.ckpt is not None:
        inr_loe.load_state_dict(torch.load(args.ckpt)) 
        start_epoch = int(args.ckpt.split('_')[-1].split('.')[0]) + 1
        logging.info("Model loaded from {}".format(args.ckpt))
        logging.info("Starting from epoch {}".format(start_epoch))
    else:
        start_epoch = 0

    # blend gating with uniform gating: alpha * uniform + (1 - alpha) * gates
    blend_alphas = [0, 0, 0, 0, 0]

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

            # iterate over the coords
            for coords, i_sel in coords_loader:
                coords, i_sel = coords.cuda(), i_sel.cuda()
                N, C, H, W = img_gt.shape
                y = img_gt.reshape(N, C, -1)[:, :, i_sel]
                y = y.permute(0, 2, 1) # N_imgs x N_coords x 3

                out, gates, importance = inr_loe(img, coords, args.top_k, args.softmax, 
                                                 blend_alphas=blend_alphas) # N_imgs x N_coords x out_dim

                mse = criterion(out, y)

                if args.top_k: 
                    loss = mse 
                else:
                    # sparsity = 0
                    variance = 0
                    for g in gates:
                        # calculate the variance of each column in the gate matrix
                        variance += torch.var(g, dim=0).mean()

                    cv_loss = []
                    if args.progressive_reverse:
                        importance = importance[::-1]
                    for l, v in enumerate(importance):
                        cv_loss.append(cv_squared_loss(v))
                        if args.progressive_epoch is not None and l == epoch // args.progressive_epoch:
                            # only calculate the cv loss for active layers
                            break
                    cv_loss = torch.stack(cv_loss).mean()
                    
                    loss = mse + cv_loss * args.cv_loss
                # compute psnr
                mse = mse.item()
                psnr = 10 * np.log10(1 / mse) # shape is N_imgs

                loss.backward()
                optim_net.step()
                optim_net.zero_grad()

                if not args.separate_optim:
                    optim_gates.step()
                    optim_gates.zero_grad()
            
            if args.separate_optim:
                optim_gates.step()
                optim_gates.zero_grad()

            if i % 25 == 0:
                logging.info("Epoch: {}, Iteration: {}, Loss: {:.4f}, PSNR: {:.4f}".format(
                    epoch, i, loss.item(), psnr.mean()))
                if args.wandb:
                    wandb.log({"loss": loss.item(), "psnr": psnr.mean()})

        if epoch % args.epochs_render == 0:
            inr_loe.eval()
            render(args, epoch, inr_loe, train_testloader, valid_coords, blend_alphas)
            render(args, epoch, inr_loe, testloader, valid_coords, blend_alphas, test=True)

        logging.info("Epoch: {}, Loss: {:.4f}, PSNR: {:.4f}".format(epoch, loss.item(), psnr.mean()))

        logging.info("Saving last model at epoch {}...".format(epoch))
        torch.save(inr_loe.state_dict(), os.path.join(args.save, "inr_loe_last.pt"))
        if epoch % args.epochs_save == 0:
            logging.info("Saving model...")
            torch.save(inr_loe.state_dict(), os.path.join(args.save, "inr_loe_{}.pt".format(epoch)))
        