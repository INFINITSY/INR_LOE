import os
import sys
import time
import copy
import math
import logging
import torch
import torch.nn as nn
import numpy as np
import argparse
import wandb
from datasets import CelebADataset, ShapeNet
from model import INRLoe
from sklearn.metrics import precision_score, recall_score
from utils import compute_loss, compute_latents, render

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='INR_LOE/data')
    parser.add_argument('--ckpt', type=str, default=None)

    # data loader
    parser.add_argument('--dataset', type=str, default='shapenet', help='dataset name')
    parser.add_argument('--batch_size', type=int, default=14)
    parser.add_argument('--train_subset', type=int, default=-1)
    parser.add_argument('--render_subset', type=int, default=9)
    
    # CelebA configs
    parser.add_argument('--side_length', type=int, default=64)

    # ShapeNet configs
    parser.add_argument('--sampling', type=int, default=None)
    parser.add_argument('--random_scale', action='store_true')

    # train params
    parser.add_argument('--epochs', type=int, default=600)
    parser.add_argument('--epochs_render', type=int, default=5)
    parser.add_argument('--epochs_save', type=int, default=5)
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

    # latent configs
    parser.add_argument('--compute_latents', action='store_true', help='compute latents for stage 2')
    
    # logging and saving
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

    # log all input arguments
    for arg in vars(args):
        if arg != 'root_dir':
            logging.info(f"{arg}: {getattr(args, arg)}")

    if args.wandb:
        wandb.init(project="inr-loe")
        wandb.config.update(args)

    if args.dataset == 'celeba':
        input_dim, output_dim = 2, 3
        trainset = CelebADataset(root=args.root_dir, split='train', subset=args.train_subset, 
                                downsampled_size=(args.side_length, args.side_length))
        train_testset = CelebADataset(root=args.root_dir, split='train', subset=args.render_subset,
                                downsampled_size=(args.side_length, args.side_length))
        testset = CelebADataset(root=args.root_dir, split='test', subset=args.render_subset,
                                downsampled_size=(args.side_length, args.side_length))
    elif args.dataset == 'shapenet':
        input_dim, output_dim = 3, 1
        trainset = ShapeNet(root=args.root_dir, split='train', sampling=args.sampling, 
                            random_scale=args.random_scale, subset=args.train_subset)
        train_testset = ShapeNet(root=args.root_dir, split='train', sampling=args.sampling, 
                            random_scale=args.random_scale, subset=args.render_subset)
        testset = ShapeNet(root=args.root_dir, split='test', sampling=args.sampling, 
                            random_scale=args.random_scale, subset=args.render_subset)
    else:
        raise ValueError("Invalid dataset")
    
    dataloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=(not args.compute_latents), num_workers=4)
    train_testloader = torch.utils.data.DataLoader(train_testset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # create the model
    inr_loe = INRLoe(input_dim=input_dim,
                     output_dim=output_dim,
                     hidden_dim=args.hidden_dim,
                     num_hidden=args.num_hidden,
                     num_exps=args.num_exps,
                     ks=args.ks,
                     latent_size=args.latent_size,
                     gate_type=args.gate_type,
                     ).cuda()

    # count the number of parameters
    params = sum(p.numel() for p in inr_loe.get_parameters())
    logging.info("Total number of parameters is: {}".format(params))
    logging.info("Model size is: {:.2f} MB".format(params * 4 / 1024**2)) 

    # create the optimizer and scheduler
    optim_net = torch.optim.Adam(inr_loe.get_parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim_net, T_max=args.epochs, eta_min=args.min_lr)
    criterion = nn.MSELoss()

    # load the model if checkpoint is provided
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

    # compute the latents for the dataset for stage 2
    if args.compute_latents:
        logging.info("Computing latents for the dataset...")
        compute_latents(args, start_epoch, inr_loe, dataloader, blend_alphas, criterion)
        # exit the program after computing the latents
        sys.exit()
    
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
        # for voxel
        acc_epoch = 0
        rec_epoch = 0
        # Outer loop: iterate over the images. 
        for i, (in_dict, gt_dict) in enumerate(dataloader):
            img = gt_dict['img'].cuda()
            idx = in_dict['idx'].cuda()
            coords = in_dict['coords'].cuda()

            if args.gate_type in ['conditional', 'separate']:
                latents = torch.randn(img.size(0), len(args.num_exps), args.latent_size).cuda() * args.std_latent
            elif args.gate_type == 'shared':
                latents = torch.randn(img.size(0), args.latent_size).cuda() * args.std_latent
            else:
                raise ValueError("Invalid gate type")
            latents.requires_grad = True

            # Initialise meta-gradient
            meta_grad = copy.deepcopy(meta_grad_init)

            if args.dataset == 'celeba':
                N, C, H, W = img.shape
                y = img.reshape(N, C, -1)
                y = y.permute(0, 2, 1) # N_imgs x N_coords x 3
            elif args.dataset == 'shapenet':
                y = img

            # Inner loop: latents update
            for _ in range(args.inner_steps):
                out, gates, importance, _ = inr_loe(latents, coords, args.top_k,
                                                blend_alphas=blend_alphas) # N_imgs x N_coords x out_dim
                loss, _ = compute_loss(args, epoch, out, y, criterion, gates, importance, 
                                        args.top_k, args.cv_loss, args.std_loss)
                latent_gradients = \
                    torch.autograd.grad(loss, latents, create_graph=True)[0]
                latents = latents - args.lr_inner * latent_gradients

            # Update the shared weights
            out, gates, importance, _ = inr_loe(latents, coords, args.top_k,
                                            blend_alphas=blend_alphas)
            loss, mse = compute_loss(args, epoch, out, y, criterion, gates, importance, 
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
            
            # compute psnr
            psnr = 10 * np.log10(1 / mse.item()) # shape is N_imgs
            psnr_epoch += psnr

            # compute acc for voxel
            acc = 0
            rec = 0
            if args.dataset == 'shapenet':
                pred = (out >= 0.5).float()
                acc = pred.eq(y).float().mean() 
                acc_epoch += acc
                rec = recall_score(y.cpu().numpy().flatten(), pred.cpu().numpy().flatten())
                rec_epoch += rec

            if i % 25 == 0:
                logging.info("Epoch: {}, Iteration: {}, Loss: {:.4f}, PSNR: {:.4f}, Acc: {:.4f}, Recall: {:.4f}".format(
                    epoch, i, loss.item(), psnr, acc, rec))
                if args.wandb:
                    wandb.log({"loss": loss.item(), "psnr": psnr, "acc": acc, "rec": rec})
        scheduler.step()

        # Render the images
        if epoch % args.epochs_render == 0:
            render(args, epoch, inr_loe, train_testloader, blend_alphas, criterion)
            render(args, epoch, inr_loe, testloader, blend_alphas, criterion, test=True)

        # Loggings and saving the model
        psnr_epoch /= len(dataloader)
        acc_epoch /= len(dataloader)
        rec_epoch /= len(dataloader)
        logging.info("Epoch: {}, PSNR: {:.4f}, Acc: {:.4f}, Recall: {:.4f}".format(
            epoch, psnr_epoch, acc_epoch, rec_epoch))
        logging.info("Saving last model at epoch {}...".format(epoch))
        if not os.path.exists(os.path.join(args.save, "ckpt")):
            os.makedirs(os.path.join(args.save, "ckpt"))
        torch.save(inr_loe.state_dict(), os.path.join(args.save, "ckpt", "inr_loe_last.pt"))
        if epoch % args.epochs_save == 0:
            logging.info("Saving model...")
            torch.save(inr_loe.state_dict(), os.path.join(args.save, "ckpt", "inr_loe_{}.pt".format(epoch)))
        