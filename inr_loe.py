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

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='data/celeba64')
    parser.add_argument('--ckpt', type=str, default=None)

    # data loader
    parser.add_argument('--train_subset', type=int, default=500)
    parser.add_argument('--test_subset', type=int, default=10)
    parser.add_argument('--side_length', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--chunk_size', type=int, default=1024)
    parser.add_argument('--render_size', type=int, default=4)

    # train params
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--epochs_render', type=int, default=10)
    parser.add_argument('--epochs_save', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--cv_loss', type=float, default=0.01, help='weight for cv loss')

    # model configs
    parser.add_argument('--top_k', action='store_true', help='whether to use top k sparce gates')
    parser.add_argument('--softmax', action='store_true', help='whether to use softmax on gates')
    parser.add_argument('--bias', action='store_true', help='use bias on weighted experts')
    parser.add_argument('--merge_before_act', action='store_true', help='merge experts before nl act')
    parser.add_argument('--num_exps', nargs='+', type=int, default=[8, 16, 64, 256, 1024])
    parser.add_argument('--ks', nargs='+', type=int, default=[4, 4, 32, 32, 256])
    parser.add_argument('--progressive_epoch', type=int, default=15, help='progressively enable experts for each layer')
    parser.add_argument('--progressive_reverse', action='store_true', help='reverse the progressive enablement')
    
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
    dataset = CelebADataset(root=args.root_dir, split='train', subset=args.train_subset, 
                            downsampled_size=(args.side_length, args.side_length))
    testset = CelebADataset(root=args.root_dir, split='test', subset=args.test_subset,
                            downsampled_size=(args.side_length, args.side_length))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    train_testloader = torch.utils.data.DataLoader(dataset, batch_size=args.render_size, shuffle=False, num_workers=4)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.render_size, shuffle=False, num_workers=4)

    coordset = LatticeDataset(image_shape=(args.side_length, args.side_length))
    coords_loader = torch.utils.data.DataLoader(coordset, pin_memory=True, batch_size=args.chunk_size, shuffle=True)

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
            img = img.cuda() # N_imgs x 3 x 32 x 32
            idx = idx.cuda() # N_imgs
            # N_imgs = img.shape[0]

            # iterate over the coords
            for coords, i_sel in coords_loader:
                coords, i_sel = coords.cuda(), i_sel.cuda()
                B, C, H, W = img.shape
                y = img.reshape(B, C, -1)[:, :, i_sel]
                y = y.permute(0, 2, 1) # N_imgs x N_coords x 3

                out, gates, importance = inr_loe(img, coords, args.top_k, args.softmax, 
                                                 blend_alphas=blend_alphas) # N_imgs x N_coords x out_dim

                mse = criterion(out, y)

                if args.top_k: 
                    loss = mse #+ sparsity#+ entropy_loss * 0.1
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

                optim_net.zero_grad()
                optim_gates.zero_grad()
                loss.backward()
                optim_net.step()
                optim_gates.step()


            if i % 25 == 0:
                logging.info("Epoch: {}, Iteration: {}, Loss: {:.4f}, PSNR: {:.4f}".format(
                    epoch, i, loss.item(), psnr.mean()))
                if args.wandb:
                    wandb.log({"loss": loss.item(), "psnr": psnr.mean()})

        if epoch % args.epochs_render == 0:
            inr_loe.eval()
            for img, idx in train_testloader:
                img = img.cuda()
                idx = idx.cuda()
                N = img.shape[0]
                coords = get_mgrid(args.side_length).cuda()
                # split coords into chunks of 128
                coords = coords.view(-1, 128, 2) 
                with torch.no_grad():
                    for i in range(0, coords.shape[0]):
                        out, gates, _ = inr_loe(img, coords[i],
                                                blend_alphas=blend_alphas) 
                        if i == 0:
                            out_all = out
                        else:
                            out_all = torch.cat((out_all, out), dim=1)
                    # out, gates, _ = inr_loe(img, coords)
                    
                    # display the output image
                    # out = out.view(N, args.side_length, args.side_length, 3)
                    out = out_all.view(N, args.side_length, args.side_length, 3)
                    # clip the output to [0, 1]
                    out = torch.clamp(out, 0, 1)
                    img = img.permute(0, 2, 3, 1)
                    # display img and out side by side
                    fig, axes = plt.subplots(N, 2, figsize=(2, 5))
                    for i in range(N):
                        axes[i, 0].imshow(img[i].cpu())
                        axes[i, 1].imshow(out[i].cpu())
                    plt.savefig(os.path.join(args.save, "output_train_e_{}.png".format(epoch)))

                    gates = [gate.detach().cpu().numpy() for gate in gates]
                    # visualize the gates
                    fig, axes = plt.subplots(1, len(gates), figsize=(15, 5))
                    for j, gate in enumerate(gates):
                        # diaplay matrix
                        axes[j].imshow(gate)
                    # save the figure
                    plt.savefig(os.path.join(args.save, "gates_train_e_{}.png".format(epoch)))

                    break
            for img, idx in testloader:
                img = img.cuda()
                idx = idx.cuda()
                N = img.shape[0]
                coords = get_mgrid(args.side_length).cuda()
                # split coords into chunks of 128
                coords = coords.view(-1, 128, 2) 
                with torch.no_grad():
                    for i in range(0, coords.shape[0]):
                        out, gates, _ = inr_loe(img, coords[i],
                                                blend_alphas=blend_alphas) 
                        if i == 0:
                            out_all = out
                        else:
                            out_all = torch.cat((out_all, out), dim=1)
                    
                    # display the output image
                    out = out_all.view(N, args.side_length, args.side_length, 3)
                    # clip the output to [0, 1]
                    out = torch.clamp(out, 0, 1)
                    img = img.permute(0, 2, 3, 1)
                    # display img and out side by side
                    fig, axes = plt.subplots(N, 2, figsize=(2, 5))
                    for i in range(N):
                        axes[i, 0].imshow(img[i].cpu())
                        axes[i, 1].imshow(out[i].cpu())
                    plt.savefig(os.path.join(args.save, "output_test_e_{}.png".format(epoch)))
                    break
            # close the figure
            plt.close('all')

        logging.info("Epoch: {}, Loss: {:.4f}, PSNR: {:.4f}".format(epoch, loss.item(), psnr.mean()))

        logging.info("Saving last model at epoch {}...".format(epoch))
        torch.save(inr_loe.state_dict(), os.path.join(args.save, "inr_loe_last.pt"))
        if epoch % args.epochs_save == 0:
            logging.info("Saving model...")
            torch.save(inr_loe.state_dict(), os.path.join(args.save, "inr_loe_{}.pt".format(epoch)))

