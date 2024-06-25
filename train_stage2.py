# ---------------------------------------------------------------
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for NVAE. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import argparse
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torchvision
from torch.cuda.amp import GradScaler, autocast
from torch.multiprocessing import Process

import stage2.utils as utils
import wandb
from datasets import LatentsDataset, get_mgrid, get_mgrid_voxel
from stage1.model import INRLoe
from stage2.nvae import AutoEncoder
from stage2.thirdparty.adamax import Adamax
from stage2.vae_backbone import LayerVAE

# from fid.fid_score import compute_statistics_of_generator, load_statistics, calculate_frechet_distance
# from fid.inception import InceptionV3


def main(args):
    # ensures that weight initializations are all the same
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    logging = utils.Logger(args.global_rank, args.save)
    # init wandb
    if args.wandb:
        wandb.init(project='stage2_nvae', config=args)
    # Get data loaders.
    # train_queue, valid_queue, num_classes = datasets.get_loaders(args)
    trainset = LatentsDataset(args.latent_path, split='train', flat=args.flat, subset=args.train_subset)
    validset = LatentsDataset(args.latent_path, split='test', flat=args.flat, subset=args.valid_subset)

    train_queue = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    valid_queue = torch.utils.data.DataLoader(validset, batch_size=args.batch_size, shuffle=False)

    args.gate_layer, args.gate_dim = trainset.latent_size

    args.num_total_iter = len(train_queue) * args.epochs
    warmup_iters = len(train_queue) * args.warmup_epochs
    swa_start = len(train_queue) * (args.epochs - 1)

    arch_instance = utils.get_arch_cells(args.arch_instance)

    model = AutoEncoder(args, arch_instance)
    # model = LayerVAE(latent_dim=args.gate_dim)
    model = model.cuda()

    logging.info('args = %s', args)
    logging.info('param size = %fM ', utils.count_parameters_in_M(model))
    logging.info('groups per scale: %s, total_groups: %d', model.groups_per_scale, sum(model.groups_per_scale))

    if args.fast_adamax:
        # Fast adamax has the same functionality as torch.optim.Adamax, except it is faster.
        cnn_optimizer = Adamax(model.parameters(), args.learning_rate,
                               weight_decay=args.weight_decay, eps=1e-3)
    else:
        # cnn_optimizer = torch.optim.Adamax(model.parameters(), args.learning_rate,
        #                                    weight_decay=args.weight_decay, eps=1e-3)
        cnn_optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate,
                                         weight_decay=args.weight_decay, eps=1e-3)

    cnn_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        cnn_optimizer, float(args.epochs - args.warmup_epochs -1), eta_min=args.learning_rate)

    # set up the INR_LOE model
    if args.pred_type == 'image':
        input_dim, output_dim = 2, 3
    elif args.pred_type == 'voxel':
        input_dim, output_dim = 3, 1
    elif args.pred_type == 'scene':
        input_dim, output_dim = 3, 1
    else:
        raise NotImplementedError

    inr_loe = INRLoe(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=64,
        num_hidden=4,
        num_exps=[64, 64, 64, 64, 64],
        ks=[8, 8, 8, 8, 8],
        latent_size=64,
        gate_type='separate',
    ).cuda()
    inr_loe.load_state_dict(torch.load(args.inr_ckpt))

    # if load
    checkpoint_file = os.path.join(args.save, 'checkpoint.pt')
    if args.cont_training:
        logging.info('loading the model.')
        checkpoint = torch.load(checkpoint_file, map_location='cpu')
        init_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        model = model.cuda()
        cnn_optimizer.load_state_dict(checkpoint['optimizer'])
        cnn_scheduler.load_state_dict(checkpoint['scheduler'])
        global_step = checkpoint['global_step']
    else:
        global_step, init_epoch = 0, 0

    for epoch in range(init_epoch, args.epochs):
        # update lrs.
        if args.distributed:
            train_queue.sampler.set_epoch(global_step + args.seed)
            valid_queue.sampler.set_epoch(0)

        if epoch > args.warmup_epochs:
            cnn_scheduler.step()

        # Logging.
        logging.info('epoch %d', epoch)

        # Training.
        train_nelbo, train_mse, global_step = train(
            epoch, train_queue, model, cnn_optimizer, global_step, warmup_iters, logging
        )
        logging.info('train_nelbo epoch %f', train_nelbo)
        logging.info('train_mse epoch %f', train_mse)
        if args.wandb:
            wandb.log({"train/nelbo_epoch": train_nelbo, "train/mse_epoch": train_mse})

        model.eval()
        # generate samples less frequently
        eval_freq = 1 if args.epochs <= 50 else 50
        if epoch % eval_freq == 0 or epoch == (args.epochs - 1):
            with torch.no_grad():
                num_samples = 16
                for t in [0.7, 0.8, 0.9, 1.0]:
                    sample, _ = model.sample(num_samples, t)
                    sample = sample.squeeze()
                    sample = sample * trainset.std + trainset.mean
                    render(args, epoch, inr_loe, sample, t)

            valid_neg_log_p, valid_nelbo, valid_mse = test(
                valid_queue, model, num_samples=10, args=args, logging=logging
            )
            logging.info("valid_nelbo %f", valid_nelbo)
            logging.info("valid neg log p %f", valid_neg_log_p)
            logging.info("valid mse %f", valid_mse)
            # logging.info('valid bpd elbo %f', valid_nelbo * bpd_coeff)
            # logging.info('valid bpd log p %f', valid_neg_log_p * bpd_coeff)

            valid_metric = {
                'val/neg_log_p': valid_neg_log_p,
                'val/nelbo': valid_nelbo,
                # 'val/bpd_log_p': valid_neg_log_p * bpd_coeff,
                # 'val/bpd_elbo': valid_nelbo * bpd_coeff
            }
            if args.wandb:
                wandb.log(valid_metric)

        save_freq = int(np.ceil(args.epochs / 100))
        if epoch % save_freq == 0 or epoch == (args.epochs - 1):
            if args.global_rank == 0:
                logging.info("saving the model.")
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "state_dict": model.state_dict(),
                        "optimizer": cnn_optimizer.state_dict(),
                        "global_step": global_step,
                        "args": args,
                        "arch_instance": arch_instance,
                        "scheduler": cnn_scheduler.state_dict(),
                    },
                    checkpoint_file,
                )

    # Final validation
    valid_neg_log_p, valid_nelbo, valid_mse = test(
        valid_queue, model, num_samples=10, args=args, logging=logging
    )
    logging.info('final valid nelbo %f', valid_nelbo)
    logging.info('final valid neg log p %f', valid_neg_log_p)
    logging.info('final valid mse %f', valid_mse)

    valid_metric = {
        'val/neg_log_p': valid_neg_log_p,
        'val/nelbo': valid_nelbo,
        # 'val/bpd_log_p': valid_neg_log_p * bpd_coeff,
        # 'val/bpd_elbo': valid_nelbo * bpd_coeff
    }
    if args.wandb:
        wandb.log(valid_metric)

    model.eval()
    with torch.no_grad():
        sample, zs = model.sample(16, t=1)
        sample = sample.squeeze()
        sample = sample * trainset.std + trainset.mean

    # render the original samples
    render(args, epoch, inr_loe, sample)

    sample_layer = args.gate_layer  # + 1 # additional layer at the first latent
    for layer in range(sample_layer):
        for idx in range(5):
            fix_z = zs[: layer + 1] + [None] * (sample_layer - layer - 1)
            with torch.no_grad():
                sample, _ = model.sample(16, t=1, fix_z=fix_z)
                sample = sample * trainset.std + trainset.mean
            render(args, epoch, inr_loe, sample, t=1, fix_layer=layer, idx=idx)


def render(args, epoch, inr_loe, sample, t=1, fix_layer=-1, idx=-1):
    inr_loe.eval()
    blend_alphas = [0] * 5
    save_path = os.path.join(args.save, 'renders', f"epoch_{epoch}")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if args.pred_type == 'image':
        coords = get_mgrid(64).cuda()
        with torch.no_grad():
            out, _, _, _ = inr_loe(sample, coords, False, blend_alphas=blend_alphas)  # N_imgs x N_coords x out_dim
        N, _, C = out.shape
        out = out.view(N, 64, 64, C).permute(0, 3, 1, 2)
        out = torch.clamp(out, 0, 1)
        grid_samples = torchvision.utils.make_grid(out, nrow=int(math.sqrt(N)))
        torchvision.utils.save_image(grid_samples, 
                                     os.path.join(save_path, f'nvae_test_e{epoch}_t{t}_{fix_layer}_{idx}.png'))

    elif args.pred_type == 'voxel':
        coords = get_mgrid_voxel(64).cuda()
        with torch.no_grad():
            out, _, _, _ = inr_loe(sample, coords, False, blend_alphas=blend_alphas)
        N = out.shape[0]
        # find the valid coords (out >= 0.5)
        valid = (out >= 0.5).float().squeeze()
        valid_coords = [coords[valid[i].bool()].cpu().numpy() for i in range(N)]
        # plot the 3D scatter plot
        fig = plt.figure()
        M = int(math.sqrt(N))
        for i, points in enumerate(valid_coords):
            ax = fig.add_subplot(M, M, i + 1, projection='3d')
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
            ax.set_box_aspect([1, 1, 1])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([min_lim, max_lim])
            ax.grid(False)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"nvae_test_e{epoch}_t{t}_{fix_layer}_{idx}.png"), 
                    bbox_inches='tight', dpi=300)
        
        plt.close(fig)

    elif args.pred_type == 'scene':
        coords = get_mgrid(64).cuda()


def train(epoch, train_queue, model, cnn_optimizer, global_step, warmup_iters, logging):
    alpha_i = utils.kl_balancer_coeff(num_scales=model.num_latent_scales,
                                      groups_per_scale=model.groups_per_scale, fun='square')
    nelbo = utils.AvgrageMeter()
    mse = utils.AvgrageMeter()
    model.train()
    for step, x in enumerate(train_queue):
        x = x[0] if len(x) > 1 else x
        x = x.cuda()

        # change bit length
        # x = utils.pre_process(x, args.num_x_bits)

        # warm-up lr
        if global_step < warmup_iters:
            lr = args.learning_rate * float(global_step) / warmup_iters
            for param_group in cnn_optimizer.param_groups:
                param_group["lr"] = lr

        # sync parameters, it may not be necessary
        if step % 100 == 0:
            utils.average_params(model.parameters(), args.distributed)

        cnn_optimizer.zero_grad()
        with autocast():
            logits, kl_all, kl_diag, log_q, log_p = model(x)
            logits = logits.squeeze()
            # output = model.decoder_output(logits)
            # kl_coeff = utils.kl_coeff(global_step, args.kl_anneal_portion * args.num_total_iter,
            #                           args.kl_const_portion * args.num_total_iter, args.kl_const_coeff)
            # kl_coeff = utils.kl_coeff_cycle_linear(global_step, args.num_total_iter, args.kl_const_coeff, args.kl_max_coeff,
            #                                        n_cycle=args.kl_cycle, ratio=args.kl_anneal_portion, constant_ratio=args.kl_const_portion)
            if epoch % 2 < 1:
                kl_coeff = args.kl_max_coeff * (step + epoch % 1 * len(train_queue)) / (1 * len(train_queue))
            elif epoch % 2 < 2:
                kl_coeff = args.kl_max_coeff
            # recon_loss = utils.reconstruction_loss(output, x, crop=model.crop_output)
            recon_loss = utils.mse_loss(logits, x)
            # recon_loss = ((logits-x)**2).sum(dim=(1,2))
            recon_loss = recon_loss.mean()

            # get kld loss
            kl_all = torch.stack(kl_all, dim=1)
            kl_coeff_i, kl_vals = utils.kl_per_group(kl_all)
            total_kl = torch.sum(kl_coeff_i)

            kl_coeff_i = kl_coeff_i / (1.0 * total_kl)
            kl_coeff_i = kl_coeff_i / torch.mean(kl_coeff_i, dim=1, keepdim=True)
            # kld_loss = torch.mean(kl_all * (kl_coeff_i.detach()))
            kld_loss = torch.sum(kl_all * (kl_coeff_i.detach()), dim=1)
            kld_loss = kld_loss.mean()
            kl_coeffs = kl_coeff_i.squeeze(0)

            # balanced_kl, kl_coeffs, kl_vals = utils.kl_balancer(kl_all, kl_coeff, kl_balance=True, alpha_i=alpha_i)
            # nelbo_batch = recon_loss + balanced_kl
            nelbo_batch = recon_loss + kld_loss * kl_coeff
            
            mse_loss = torch.mean(recon_loss)
            loss = torch.mean(nelbo_batch)
            # norm_loss = model.spectral_norm_parallel()
            # bn_loss = model.batchnorm_loss()
            # get spectral regularization coefficient (lambda)
            # if args.weight_decay_norm_anneal:
            #     assert args.weight_decay_norm_init > 0 and args.weight_decay_norm > 0, 'init and final wdn should be positive.'
            #     wdn_coeff = (1. - kl_coeff) * np.log(args.weight_decay_norm_init) + kl_coeff * np.log(args.weight_decay_norm)
            #     wdn_coeff = np.exp(wdn_coeff)
            # else:
            #     wdn_coeff = args.weight_decay_norm

            # loss += norm_loss * wdn_coeff + bn_loss * wdn_coeff

        # grad_scalar.scale(loss).backward()
        loss.backward()
        utils.average_gradients(model.parameters(), args.distributed)
        # grad_scalar.step(cnn_optimizer)
        cnn_optimizer.step()
        # grad_scalar.update()
        nelbo.update(loss.data, 1)
        # mse.update(mse_loss.data, 1)

        if (global_step + 1) % 100 == 0:
            # if (global_step + 1) % 1000 == 0:  # reduced frequency
            #     n = int(np.floor(np.sqrt(x.size(0))))
            #     x_img = x[:n*n]
            #     output_img = output.mean if isinstance(output, torch.distributions.bernoulli.Bernoulli) else output.sample()
            #     output_img = output_img[:n*n]
            #     x_tiled = utils.tile_image(x_img, n)
            #     output_tiled = utils.tile_image(output_img, n)
            #     in_out_tiled = torch.cat((x_tiled, output_tiled), dim=2)
            #     writer.add_image('reconstruction', in_out_tiled, global_step)

            # norm
            # writer.add_scalar('train/norm_loss', norm_loss, global_step)
            # writer.add_scalar('train/bn_loss', bn_loss, global_step)
            # writer.add_scalar('train/norm_coeff', wdn_coeff, global_step)

            utils.average_tensor(nelbo.avg, args.distributed)
            utils.average_tensor(mse.avg, args.distributed)
            logging.info("train %d %f %f", global_step, nelbo.avg, mse.avg)

            # writer.add_scalar('train/recon_iter', torch.mean(utils.reconstruction_loss(output, x, crop=model.crop_output)), global_step)
            train_metric = {
                "train/nelbo_avg": nelbo.avg,
                "train/mse_avg": mse.avg,
                "train/lr": cnn_optimizer.state_dict()["param_groups"][0]["lr"],
                "train/nelbo_iter": loss,
                "train/kl_iter": torch.mean(torch.sum(kl_all, dim=1)),
                "train/mse_iter": mse_loss,
                "kl_coeff/coeff": kl_coeff,
            }
            total_active = 0
            for i, kl_diag_i in enumerate(kl_diag):
                utils.average_tensor(kl_diag_i, args.distributed)
                num_active = torch.sum(kl_diag_i > 0.1).detach()
                total_active += num_active

                # kl_ceoff
                # update the metrics
                train_metric[f"kl/active_{i}"] = num_active
                train_metric[f"kl_coeff/layer_{i}"] = kl_coeffs[i]
                train_metric[f"kl_vals/layer_{i}"] = kl_vals[i]
            # writer.add_scalar('kl/total_active', total_active, global_step)
            train_metric["kl/total_active"] = total_active

            if args.wandb:
                wandb.log(train_metric)

        global_step += 1

    utils.average_tensor(nelbo.avg, args.distributed)
    utils.average_tensor(mse.avg, args.distributed)
    return nelbo.avg, mse.avg, global_step


def test(valid_queue, model, num_samples, args, logging):
    if args.distributed:
        dist.barrier()
    nelbo_avg = utils.AvgrageMeter()
    neg_log_p_avg = utils.AvgrageMeter()
    mse_avg = utils.AvgrageMeter()
    model.eval()
    for step, x in enumerate(valid_queue):
        x = x[0] if len(x) > 1 else x
        x = x.cuda()

        # change bit length
        # x = utils.pre_process(x, args.num_x_bits)

        with torch.no_grad():
            nelbo, log_iw, mse = [], [], 0
            for k in range(num_samples):
                logits, kl_all, _, log_q, log_p = model(x)
                logits = logits.squeeze()
                # output = model.decoder_output(logits)
                # recon_loss = utils.reconstruction_loss(output, x, crop=model.crop_output)
                recon_loss = utils.mse_loss(logits, x)
                # recon_loss = ((logits-x)**2).sum(dim=(1,2))
                recon_loss = recon_loss.mean()

                # balanced_kl, _, _ = utils.kl_balancer(kl_all, kl_balance=False)
                # nelbo_batch = recon_loss + balanced_kl

                # get kld loss
                kl_all = torch.stack(kl_all, dim=1)
                # kl_coeff_i, kl_vals = utils.kl_per_group(kl_all)
                # total_kl = torch.sum(kl_coeff_i)

                # kl_coeff_i = kl_coeff_i / (1.0 * total_kl)
                # kl_coeff_i = kl_coeff_i / torch.mean(kl_coeff_i, dim=1, keepdim=True)
                # kld_loss = torch.mean(kl_all * (kl_coeff_i.detach()))
                # kld_loss = torch.sum(kl_all * (kl_coeff_i.detach()), dim=1)
                kld_loss = torch.sum(kl_all, dim=1)
                kld_loss = kld_loss.mean()
                # kl_coeffs = kl_coeff_i.squeeze(0)

                nelbo_batch = recon_loss + kld_loss
                nelbo.append(nelbo_batch)
                mse += recon_loss
                # log_iw.append(utils.log_iw(output, x, log_q, log_p, crop=model.crop_output))
                log_iw.append(utils.log_iw_mse(logits, x, log_q, log_p))

            # nelbo = torch.mean(torch.stack(nelbo, dim=1))
            nelbo = torch.mean(torch.stack(nelbo))
            log_p = torch.mean(torch.logsumexp(torch.stack(log_iw, dim=1), dim=1) - np.log(num_samples))
            mse = mse / num_samples

        nelbo_avg.update(nelbo.data, x.size(0))
        neg_log_p_avg.update(-log_p.data, x.size(0))
        mse_avg.update(mse.data, x.size(0))

    utils.average_tensor(nelbo_avg.avg, args.distributed)
    utils.average_tensor(neg_log_p_avg.avg, args.distributed)
    utils.average_tensor(mse_avg.avg, args.distributed)
    if args.distributed:
        # block to sync
        dist.barrier()
    logging.info('val, step: %d, NELBO: %f, neg Log p %f, mse: %f', step, nelbo_avg.avg, neg_log_p_avg.avg, mse_avg.avg)
    return neg_log_p_avg.avg, nelbo_avg.avg, mse_avg.avg


def create_generator_vae(model, batch_size, num_total_samples):
    num_iters = int(np.ceil(num_total_samples / batch_size))
    for i in range(num_iters):
        with torch.no_grad():
            logits = model.sample(batch_size, 1.0)
            output = model.decoder_output(logits)
            output_img = output.mean if isinstance(output, torch.distributions.bernoulli.Bernoulli) else output.mean()
        yield output_img.float()


# def test_vae_fid(model, args, total_fid_samples):
#     dims = 2048
#     device = 'cuda'
#     num_gpus = args.num_process_per_node * args.num_proc_node
#     num_sample_per_gpu = int(np.ceil(total_fid_samples / num_gpus))

#     g = create_generator_vae(model, args.batch_size, num_sample_per_gpu)
#     block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
#     model = InceptionV3([block_idx], model_dir=args.fid_dir).to(device)
#     m, s = compute_statistics_of_generator(g, model, args.batch_size, dims, device, max_samples=num_sample_per_gpu)

#     # share m and s
#     m = torch.from_numpy(m).cuda()
#     s = torch.from_numpy(s).cuda()
#     # take average across gpus
#     utils.average_tensor(m, args.distributed)
#     utils.average_tensor(s, args.distributed)

#     # convert m, s
#     m = m.cpu().numpy()
#     s = s.cpu().numpy()

#     # load precomputed m, s
#     path = os.path.join(args.fid_dir, args.dataset + '.npz')
#     m0, s0 = load_statistics(path)

#     fid = calculate_frechet_distance(m0, s0, m, s)
#     return fid


def init_processes(rank, size, fn, args):
    """Initialize the distributed environment."""
    os.environ["MASTER_ADDR"] = args.master_address
    os.environ["MASTER_PORT"] = "6021"
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=size)
    fn(args)
    cleanup()


def cleanup():
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("encoder decoder examiner")
    # experimental results
    parser.add_argument('--root', type=str, default='loe_nvae/exp',
                        help='location of the results')
    parser.add_argument('--save', type=str, default='stage2',
                        help='id used for storing intermediate results')
    # data
    parser.add_argument('--dataset', type=str, default='celeba_64',
                        choices=['celeba', 'shapenet', 'snrcars'],
                        help='which dataset to use')
    parser.add_argument('--data', type=str, default='/tmp/nasvae/data',
                        help='location of the data corpus')
    parser.add_argument('--pred_type', type=str, default='voxel',
                        help='type of prediction. Options: image, voxel, scene')
    # optimization
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch size per GPU')
    parser.add_argument('--learning_rate', type=float, default=1e-2,
                        help='init learning rate')
    parser.add_argument('--learning_rate_min', type=float, default=1e-3,
                        help='min learning rate')
    parser.add_argument('--weight_decay', type=float, default=3e-4,
                        help='weight decay')
    parser.add_argument('--weight_decay_norm', type=float, default=0.,
                        help='The lambda parameter for spectral regularization.')
    parser.add_argument('--weight_decay_norm_init', type=float, default=10.,
                        help='The initial lambda parameter')
    parser.add_argument('--weight_decay_norm_anneal', action='store_true', default=False,
                        help='This flag enables annealing the lambda coefficient from '
                             '--weight_decay_norm_init to --weight_decay_norm.')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='num of training epochs')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='num of training epochs in which lr is warmed up')
    parser.add_argument('--fast_adamax', action='store_true', default=False,
                        help='This flag enables using our optimized adamax.')
    parser.add_argument('--arch_instance', type=str, default='res_mbconv',
                        help='path to the architecture instance')
    # KL annealing
    parser.add_argument('--kl_anneal_portion', type=float, default=0.5,
                        help='The portions epochs that KL is annealed')
    parser.add_argument('--kl_const_portion', type=float, default=0.2,
                        help='The portions epochs that KL is constant at kl_const_coeff')
    parser.add_argument('--kl_const_coeff', type=float, default=0.0001,
                        help='The constant value used for min KL coeff')
    parser.add_argument('--kl_max_coeff', type=float, default=0.0005,
                        help='The maximum value used for max KL coeff')
    parser.add_argument('--kl_cycle', type=int, default=5, 
                        help='The number of cycles for annealing')
    # Flow params
    parser.add_argument('--num_nf', type=int, default=0,
                        help='The number of normalizing flow cells per groups. Set this to zero to disable flows.')
    parser.add_argument('--num_x_bits', type=int, default=8,
                        help='The number of bits used for representing data for colored images.')
    # latent variables
    parser.add_argument('--num_latent_scales', type=int, default=1,
                        help='the number of latent scales')
    parser.add_argument('--num_groups_per_scale', type=int, default=5,
                        help='number of groups of latent variables per scale')
    parser.add_argument('--num_latent_per_group', type=int, default=20,
                        help='number of channels in latent variables per group')
    parser.add_argument('--ada_groups', action='store_true', default=False,
                        help='Settings this to true will set different number of groups per scale.')
    parser.add_argument('--min_groups_per_scale', type=int, default=1,
                        help='the minimum number of groups per scale.')
    # encoder parameters
    parser.add_argument('--num_channels_enc', type=int, default=48,
                        help='number of channels in encoder')
    parser.add_argument('--num_preprocess_blocks', type=int, default=2,
                        help='number of preprocessing blocks')
    parser.add_argument('--num_preprocess_cells', type=int, default=3,
                        help='number of cells per block')
    parser.add_argument('--num_cell_per_cond_enc', type=int, default=1,
                        help='number of cell for each conditional in encoder')
    # decoder parameters
    parser.add_argument('--num_channels_dec', type=int, default=48,
                        help='number of channels in decoder')
    parser.add_argument('--num_postprocess_blocks', type=int, default=2,
                        help='number of postprocessing blocks')
    parser.add_argument('--num_postprocess_cells', type=int, default=3,
                        help='number of cells per block')
    parser.add_argument('--num_cell_per_cond_dec', type=int, default=1,
                        help='number of cell for each conditional in decoder')
    parser.add_argument('--num_mixture_dec', type=int, default=1,
                        help='number of mixture components in decoder. set to 1 for Normal decoder.')
    # NAS
    parser.add_argument('--use_se', action='store_true', default=False,
                        help='This flag enables squeeze and excitation.')
    parser.add_argument('--res_dist', action='store_true', default=False,
                        help='This flag enables squeeze and excitation.')
    parser.add_argument('--cont_training', action='store_true', default=False,
                        help='This flag enables training from an existing checkpoint.')
    # DDP.
    parser.add_argument('--num_proc_node', type=int, default=1,
                        help='The number of nodes in multi node env.')
    parser.add_argument('--node_rank', type=int, default=0,
                        help='The index of node.')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='rank of process in the node')
    parser.add_argument('--global_rank', type=int, default=0,
                        help='rank of process among all the processes')
    parser.add_argument('--num_process_per_node', type=int, default=1,
                        help='number of gpus')
    parser.add_argument('--master_address', type=str, default='127.0.0.1',
                        help='address for master')
    parser.add_argument('--seed', type=int, default=1,
                        help='seed used for initialization')
    
    # latents specifc
    parser.add_argument('--latent_path', type=str, 
                        default='save/20240521_215600_shape_b16_l64_compute_latents',
                        help='location of the computed latents')
    parser.add_argument('--flat', action='store_true', default=False,
                        help='This flag enables flattening the latents.')
    parser.add_argument('--train_subset', type=int, default=-1,
                        help='The number of training samples to use.')
    parser.add_argument('--valid_subset', type=int, default=-1,
                        help='The number of validation samples to use.')
    
    # inr_loe specific
    parser.add_argument('--inr_ckpt', type=str, 
                        default='save/20240516_235920_shape_b16_l64/ckpt/inr_loe_335.pt',
                        help='location of the INR_LOE checkpoint')
    
    # wandb
    parser.add_argument("--wandb", type=bool, default=True, help="This flag enables logging to wandb.")

    args = parser.parse_args()
    # args.save = args.root + '/eval-' + args.save
    args.save = os.path.join(args.latent_path, args.save)
    utils.create_exp_dir(args.save)

    size = args.num_process_per_node

    if size > 1:
        args.distributed = True
        processes = []
        for rank in range(size):
            args.local_rank = rank
            global_rank = rank + args.node_rank * args.num_process_per_node
            global_size = args.num_proc_node * args.num_process_per_node
            args.global_rank = global_rank
            print("Node rank %d, local proc %d, global proc %d" % (args.node_rank, rank, global_rank))
            p = Process(target=init_processes, args=(global_rank, global_size, main, args))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:
        # for debugging
        print("starting in debug mode")
        # args.distributed = True
        args.distributed = False
        init_processes(0, size, main, args)
