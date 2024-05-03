import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import argparse
import math
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from model import CVAE, VAE
from datasets import LatentsDataset, get_mgrid
from model import INRLoe


def loss_function(recon_x, x, mu, logvar, kld_weight=0.1):
    # Reconstruction loss
    # BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    MSE = F.mse_loss(recon_x, x)
    # KL divergence
    # KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1), dim = 0)
    return MSE + KLD * kld_weight

parser = argparse.ArgumentParser()

# general configs
parser.add_argument('--side_length', type=int, default=64, help='side length of the image')
parser.add_argument('--patch_size', type=int, default=None)

# inr model configs
parser.add_argument('--ckpt', type=str, default='save/20240502_221525_separate_5_layer_b14_latent_64/ckpt/inr_loe_595.pt',
                    help='path to the model checkpoint')
parser.add_argument('--top_k', action='store_true', help='whether to use top k sparce gates')
parser.add_argument('--softmax', action='store_true', help='whether to use softmax on gates')
parser.add_argument('--bias', action='store_true', help='use bias on weighted experts')
parser.add_argument('--merge_before_act', type=bool, default=True, help='merge experts before nl act')
parser.add_argument('--num_exps', nargs='+', type=int, default=[64, 64, 64, 64, 64])
parser.add_argument('--ks', nargs='+', type=int, default=[8, 8, 8, 8, 8])
parser.add_argument('--progressive_epoch', type=int, default=None, help='progressively enable experts for each layer')
parser.add_argument('--progressive_reverse', action='store_true', help='reverse the progressive enablement')
parser.add_argument('--latent_size', type=int, default=64, help='size of the latent space')
parser.add_argument('--num_hidden', type=int, default=4, help='number of hidden layers')
parser.add_argument('--hidden_dim', type=int, default=64, help='hidden layer dim')
parser.add_argument('--std_latent', type=float, default=0.0001, help='std of latent sampling')
parser.add_argument('--gate_type', type=str, default='separate', help='gating type: separate, conditional, or shared')

# latent configs
parser.add_argument('--latent_path', type=str, default='save/20240503_123320_compute_latents_5_layer_b14_latent_64_30000/latents_train_e_596.pt')
parser.add_argument('--flat', action='store_true', help='whether the latents are flat')
parser.add_argument('--subset', type=int, default=30000, help='subset of the latents')
parser.add_argument('--layer_learn', type=int, default=1, help='layer to learn dist')

# training configs
parser.add_argument('--vae_batch_size', type=int, default=128, help='batch size')
parser.add_argument('--vae_epochs', type=int, default=1000, help='number of epochs')
parser.add_argument('--vae_lr', type=float, default=5e-3, help='learning rate')
parser.add_argument('--vae_hidden_dim', type=int, default=48, help='hidden layer dim')
parser.add_argument('--vae_latent_size', type=int, default=32, help='size of the latent space')
parser.add_argument('--vae_condition_dim', type=int, default=64, help='size of the condition vector')
parser.add_argument('--epochs_log', type=int, default=10, help='log every n epochs')
parser.add_argument('--kld_weight', type=float, default=0.001, help='weight of the kld loss')
parser.add_argument('--vae_ckpt', type=str, default=None, help='path to the vae model checkpoint')
parser.add_argument('--eval_only', action='store_true', help='only evaluate the model')

parser.add_argument('--exp_cmt', type=str, default='conditional_vae')
args = parser.parse_args()

args.save = '/'.join(args.latent_path.split('/')[:-1])
if not os.path.exists(os.path.join(args.save, args.exp_cmt)):
    os.makedirs(os.path.join(args.save, args.exp_cmt))
args.save = os.path.join(args.save, args.exp_cmt)

# load latents 
dataset = LatentsDataset(args.latent_path, flat=args.flat, subset=args.subset, layer=args.layer_learn)
# print(f'mean: {dataset.mean}, std: {dataset.std}')
dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.vae_batch_size, shuffle=True)
# Example usage

input_size = dataset.latent_size[-1]
# condition_size = dataset.latent_size[1]

# model = CVAE(input_size, condition_size, latent_size, hidden_size)
model = VAE(input_size, args.vae_latent_size, args.vae_hidden_dim, args.vae_condition_dim).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=args.vae_lr)

if args.vae_ckpt is not None:
    model.load_state_dict(torch.load(args.vae_ckpt))
    print(f'Loaded model from {args.vae_ckpt}')

if not args.eval_only:
    for epoch in range(args.vae_epochs):
        model.train()
        train_loss = 0
        for i, (data, _) in enumerate(dataloader):
            data = data.cuda()
            if len(data.size()) > 2:
                condition, input_data = data[:, 0], data[:, 1]
            else:
                condition, input_data = None, data
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(input_data, condition)
            loss = loss_function(recon_batch, input_data, mu, logvar, args.kld_weight)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        if epoch % args.epochs_log == 0:
            print(f'Epoch {epoch}, Loss {train_loss/len(dataloader)}')
        if epoch % 200 == 0:
            torch.save(model.state_dict(), 
                    os.path.join(args.save, f'vae_{epoch}.pt'))
    torch.save(model.state_dict(), 
            os.path.join(args.save, f'vae_{args.vae_epochs}.pt'))

# sample from the model
model.eval()
with torch.no_grad():
    sample = torch.randn(16, args.vae_latent_size).cuda()
    sample = model.decode(sample)
    # add back mean and std
    sample = sample * dataset.std + dataset.mean

if args.flat:
    sample = sample.view(sample.shape[0], len(args.num_exps), -1)

inr_loe = INRLoe(hidden_dim=args.hidden_dim,
                num_hidden=args.num_hidden,
                num_exps=args.num_exps,
                ks=args.ks,
                image_resolution=args.side_length, 
                merge_before_act=args.merge_before_act, 
                bias=args.bias,
                patchwise=(args.patch_size is not None),
                latent_size=args.latent_size,
                gate_type=args.gate_type,
                ).cuda()

inr_loe.load_state_dict(torch.load(args.ckpt))
inr_loe_epoch = args.ckpt.split('/')[-1].split('.')[0].split('_')[-1]
inr_loe.eval()
blend_alphas = [0] * len(inr_loe.num_exps)
coords = get_mgrid(args.side_length).cuda()

with torch.no_grad():
    out, gates, importance, _ = inr_loe(sample, coords, args.top_k, args.softmax,
                                        blend_alphas=blend_alphas) # N_imgs x N_coords x out_dim

N, N_coords, C = out.shape
out = out.view(N, args.side_length, args.side_length, C).permute(0, 3, 1, 2)
out = torch.clamp(out, 0, 1)

grid_samples = torchvision.utils.make_grid(out, nrow=int(math.sqrt(out.size(0))))
torchvision.utils.save_image(grid_samples, 
                             os.path.join(args.save, 
                                          f'vae_inr_loe_{inr_loe_epoch}_{args.vae_epochs}.png'))

# also save the train latents
# train_latents = dataset.latents
# if args.flat:
#     train_latents = train_latents.view(train_latents.shape[0], len(args.num_exps), -1)
# with torch.no_grad():
#     train_latents = train_latents.cuda()
#     train_out, _, _, _ = inr_loe(train_latents, coords, args.top_k, args.softmax,
#                                  blend_alphas=blend_alphas) # N_imgs x N_coords x out_dim
# N, N_coords, C = train_out.shape
# train_out = train_out.view(N, args.side_length, args.side_length, C).permute(0, 3, 1, 2)
# train_out = torch.clamp(train_out, 0, 1)

# grid_samples = torchvision.utils.make_grid(train_out, nrow=int(math.sqrt(train_out.size(0))))
# torchvision.utils.save_image(grid_samples, 
#                              os.path.join(args.save, 
#                                           f'train.png'))

# visualize the gates
import matplotlib.pyplot as plt

gates = [gate.detach().cpu().numpy() for gate in gates]
fig, axes = plt.subplots(1, len(gates), figsize=(16, 4))
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
plt.savefig(os.path.join(args.save, "gates_train_e_{}.png".format(epoch)), 
            bbox_inches='tight', dpi=300)


