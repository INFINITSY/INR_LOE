"""
Here we explore different backbones for inferring params: vae decoder tailored for 1d features.
"""

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

"""
For vae_backbone, we need to specify:
    1. how many layers for each encoder/decoder cell.
    2. how to be compatible with 1d features.

For a simple vae for 20 pointclouds data, we only use z (b, 128), feat (b, 1024) to generate new samples and fit kld.
To scale up to large size data like 100K samples, we need to consider:
    1. If we should increase z's size to express more data.
    2. If we should increase feat's size so as to increase siren's capacity.
As the first attempt, we will keep z's and feat's sizes fixed. We increase the encoder and decoder number of cell to implement a hierarchical structure.
Secondly, we will try to increase z's size to express more data.
Thirdly, we will try to increase feat's size to increase siren's capacity.
What is more, we have to be very careful for the sigma of prior distribution to avoid under/over-fitting to prior/data.
"""


# --------------------Component classes--------------------#
# Encoder combiner cell
class EncCombinerCell(nn.Module):
    def __init__(self, Cin2, Cout):
        super(EncCombinerCell, self).__init__()
        # Cin = Cin1 + Cin2
        self.conv = nn.Linear(Cin2, Cout, bias=True)

    def forward(self, x1, x2):
        x2 = self.conv(x2)
        out = x1 + x2
        return out


# Decoder combiner cell
class DecCombinerCell(nn.Module):
    def __init__(self, Cin1, Cin2, Cout):
        super(DecCombinerCell, self).__init__()
        self.conv = nn.Linear(Cin2 + Cin1, Cout, bias=True)

    def forward(self, x1, x2):
        out = torch.cat([x1, x2], dim=2)
        out = self.conv(out)
        return out


class ZCombiner(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()
        self.combiner = nn.Sequential(
            nn.Linear(inc, outc),
            nn.SELU(),
            nn.Linear(outc, outc),
        )

    def forward(self, z):
        s = self.combiner(z)
        mu, log_sigma = torch.chunk(s, 2, dim=2)

        return mu, log_sigma


@torch.jit.script
def soft_clamp5(x: torch.Tensor):
    return x.div(5.0).tanh_().mul(5.0)


@torch.jit.script
def sample_normal_jit(mu, sigma):
    eps = mu.mul(0).normal_()
    z = eps.mul_(sigma).add_(mu)
    return z, eps


class Normal:
    def __init__(self, mu, log_sigma, temp=1.0):
        self.mu = soft_clamp5(mu)
        log_sigma = soft_clamp5(log_sigma)
        self.sigma = torch.exp(log_sigma) + 1e-2  # we don't need this after soft clamp
        if temp != 1.0:
            self.sigma *= temp

    def sample(self):
        return sample_normal_jit(self.mu, self.sigma)

    def sample_given_eps(self, eps):
        return eps * self.sigma + self.mu

    def log_p(self, samples, eps=1e-5):
        normalized_samples = (samples - self.mu) / self.sigma
        log_p = (
            -0.5 * normalized_samples * normalized_samples
            - 0.5 * np.log(2 * np.pi)
            - torch.log(self.sigma)
        )
        return log_p

    def kl(self, normal_dist):
        term1 = (self.mu - normal_dist.mu) / normal_dist.sigma
        term2 = self.sigma / normal_dist.sigma

        return 0.5 * (term1 * term1 + term2 * term2) - 0.5 - torch.log(term2)

    def to_device(self, device):
        self.mu = self.mu.to(device)
        self.sigma = self.sigma.to(device)
        return self


# Attention cell
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


"""
Input args:
    dim: length of input letents, e.g. 1024
    depth: number of layers
    heads: number of heads in multi-head attention
    dim_head: the inner length of each head in attention module, e.g. 64
    mlp_dim: the length of output latents, e.g. 1024
"""


class AttnCell(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, out_dim, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(out_dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                        FeedForward(dim, mlp_dim, dropout=dropout),
                    ]
                )
            )
        self.to_out = nn.Linear(dim, out_dim)

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        x = self.to_out(x)
        return self.norm(x)


# --------------------Main classes--------------------#
"""
VAE BACKBONE:
    input_channel: 1, input tensor is of shape (b, 1, L) = (batch_size, input_channel, input_length)
"""


class HVAEBackbone(nn.Module):
    def __init__(
        self,
        input_channel=1,
        latent_channel=64,
        layers=[2, 2, 2, 2],
        sampler_dim=40,
        prior_scale=1e-4,
        num_heads=8,
        dim_head=64,
        dropout=0.25,
        sample_decoder=False,
        latent_dim=1024,
    ):
        super(HVAEBackbone, self).__init__()

        num_layers = len(layers)  # How many layers for encoder/decoder tower

        self.prior_scale = prior_scale  # prior sclae for generating latents: in our experience, 1e-4 is a good choice.
        self.num_layers = num_layers
        self.sample_decoder = sample_decoder  # should we sample the last layer of decoder? True: a learned std, False: a prior std of 1

        self.encoder_tower = nn.ModuleList()
        self.decoder_tower = nn.ModuleList()

        self.enc_combiners = nn.ModuleList()
        self.dec_combiners = nn.ModuleList()

        self.enc_sampler = nn.ModuleList()
        self.dec_sampler = nn.ModuleList()

        # get encoder tower
        """
        Channel:
            layer 1: 64 -> 64
            layer 2: 64 -> 128
            layer 3: 128 -> 256
            layer n > 3: 256 -> 256
        """
        for n in range(num_layers):
            if n == 0:  # first layer of encoder, channels from 64 to 64
                Cin = latent_channel
                Cout = latent_channel
            else:
                Cin = latent_channel * (np.power(2, min(n, 3) - 1))
                Cout = latent_channel * (np.power(2, min(n, 2)))
            self.encoder_tower.append(
                AttnCell(
                    dim=Cin,
                    depth=int(layers[n]),
                    heads=num_heads,
                    dim_head=dim_head,
                    mlp_dim=Cin,
                    out_dim=Cout,
                    dropout=dropout,
                )
            )
            self.enc_combiners.append(EncCombinerCell(Cout, Cout))
            self.enc_sampler.append(
                nn.Sequential(nn.Linear(Cout, sampler_dim, bias=True))
            )

        # get decoder tower
        """
        Channel:
            layer n>3: 256 -> 256
            layer 3: 256 -> 128
            layer 2: 128 -> 64
            layer 1: 64 -> 64
        """
        for n in range(num_layers):
            if n == num_layers - 1:  # last layer of decoder, channels from 64 to 64
                Cin = latent_channel
                Cout = latent_channel
            else:
                Cin = latent_channel * (2 ** (min(num_layers - n - 1, 2)))
                Cout = latent_channel * (2 ** (min(num_layers - n - 2, 2)))
            self.decoder_tower.append(
                AttnCell(
                    dim=Cin,
                    depth=int(layers[n]),
                    heads=num_heads,
                    dim_head=dim_head,
                    mlp_dim=Cin,
                    out_dim=Cout,
                    dropout=dropout,
                )
            )
            self.dec_combiners.append(DecCombinerCell(sampler_dim // 2, Cin, Cin))
            if (
                n < num_layers - 1
            ):  # As the NVAE, the number of decoder equals the number of encoder minus 1
                self.dec_sampler.append(
                    nn.Sequential(nn.Linear(Cout, sampler_dim, bias=True))
                )

        self.enc_0 = nn.Sequential(
            nn.ELU(),
            nn.Linear(
                in_features=latent_channel * (2**num_layers),
                out_features=latent_channel * (2**num_layers),
            ),
        )

        """
        For a tensor with shape (b,c,l), we firstly reshape to (b, c, l1, l2)
        """
        ldmp = self.get_maximal_factor_pair(latent_dim)
        lcmp = self.get_maximal_factor_pair(latent_channel)
        self.pre_processor = nn.Sequential(
            nn.Conv1d(
                in_channels=input_channel,
                out_channels=latent_channel,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ),
            Rearrange(
                "b (l1 l2) (l3 l4) -> b (l1 l3) (l2 l4)",
                l1=lcmp[0],
                l2=lcmp[1],
                l3=ldmp[0],
                l4=ldmp[1],
            ),
            nn.Conv1d(
                in_channels=lcmp[0] * ldmp[0],
                out_channels=latent_channel,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ),
            Rearrange("b c l -> b l c", c=latent_channel),
        )
        # to generate prior z_1
        prior_ftr0_size = (
            lcmp[1] * ldmp[1],
            (2 ** (min(num_layers - 1, 2))) * latent_channel,
        )  # should align with the siren latents' length
        self.prior_ftr0 = nn.Parameter(
            torch.rand(size=prior_ftr0_size), requires_grad=True
        )
        self.post_processor = nn.Sequential(
            nn.Conv1d(
                lcmp[1] * ldmp[1],
                latent_dim,
                kernel_size=1,
                padding=0,
                stride=1,
                bias=True,
            ),
            nn.ELU(),
            nn.Linear(latent_channel, input_channel, bias=True),
        )
        if sample_decoder:
            self.dist_embedding = ZCombiner(
                latent_dim, 2 * latent_dim
            )  # should align with the inr latents' length

    def get_maximal_factor_pair(self, number):
        for i in range(1, int(number**0.5) + 1):
            if number % i == 0:
                factor_pair = (i, number // i)
                max_factor_pair = factor_pair
        return max_factor_pair

    def forward(self, z_in, return_meta=False):
        # s1. we want to get posterior zn,...,z2,z1,z0 -> z0,z1,z2,...,zn
        z_n = self.pre_processor(z_in)
        z_posterior = []
        z_enc_combiners = []
        for n in range(self.num_layers):
            z_n = self.encoder_tower[n](z_n)  # (b, L, 64->64->128->256)
            z_posterior.append(z_n)
            z_enc_combiners.append(self.enc_combiners[n])
        z_posterior.reverse()
        z_enc_combiners.reverse()

        # s2. get the first posterior
        param_0 = self.enc_0[0](z_posterior[0])  # q_0 (b, L, 256)
        param_0 = self.enc_sampler[self.num_layers - 1](param_0)  # q_0 (b, L, 40)
        mu_q, log_sigma_q = torch.chunk(
            param_0, 2, dim=2
        )  # mu_q, log_sigma_q (b, L,20)
        dist = Normal(mu_q, log_sigma_q)
        z, _ = dist.sample()  # z (b, L, 20)
        log_q = dist.log_p(z)  # (b, L, 20)
        all_q = [dist]
        all_log_q = [log_q]

        # s3. get the first prior
        dist = Normal(
            torch.zeros_like(mu_q),
            torch.zeros_like(log_sigma_q)
            + torch.log(torch.tensor(self.prior_scale, device=mu_q.device)),
        )
        log_p = dist.log_p(z)  # (b, L, 20)
        all_p = [dist]
        all_log_p = [log_p]

        """
        s is a random input sample, It doesn't affect the kld of p and q.
        For further exploration: s can be class label, hierarchy cluster.
        """
        s = self.prior_ftr0.unsqueeze(0)
        batch_size = z.size(0)
        s = s.expand(batch_size, -1, -1)
        s = self.dec_combiners[0](s, z)
        s = self.decoder_tower[0](s)

        # s4. get the combined posterior from the features of generative model
        for n in range(1, self.num_layers):
            # a1. get prior
            param = self.dec_sampler[n - 1](s)
            mu_p, log_sigma_p = torch.chunk(param, 2, dim=2)

            # a2. get posterior
            ftr = z_enc_combiners[n](z_posterior[n], s)
            param = self.enc_sampler[self.num_layers - n - 1](ftr)
            mu_q, log_sigma_q = torch.chunk(param, 2, dim=2)
            dist_q = Normal(mu_p + mu_q, log_sigma_p + log_sigma_q)
            # dist_q = Normal(mu_q, log_sigma_q)
            z, _ = dist_q.sample()
            log_q = dist_q.log_p(z)
            all_q.append(dist_q)
            all_log_q.append(log_q)

            dist_p = Normal(mu_p, log_sigma_p)
            log_p = dist_p.log_p(z)
            all_p.append(dist_p)
            all_log_p.append(log_p)

            # a3. combine posterior with prior features
            s = self.dec_combiners[n](s, z)
            s = self.decoder_tower[n](s)

        # the last combined s is sent to post_processor
        s = self.post_processor(s)

        # Sample decoder or not: True for a learned std, False for a prior std of 1
        if self.sample_decoder:
            z_mu, z_sigma = self.dist_embedding(s)
            out_dist = Normal(z_mu, z_sigma)
        else:
            out_dist = s

        kl_all = []
        kl_diag = []
        log_p, log_q = 0.0, 0.0
        for q, p, log_q_conv, log_p_conv in zip(all_q, all_p, all_log_q, all_log_p):
            kl_per_var = q.kl(p)

            kl_diag.append(torch.mean(torch.sum(kl_per_var, dim=[1, 2]), dim=0))
            kl_all.append(torch.sum(kl_per_var, dim=[1, 2]))
            log_q += torch.sum(log_q_conv, dim=[1, 2])
            log_p += torch.sum(log_p_conv, dim=[1, 2])
        if return_meta:
            all_log_q = [torch.sum(log_q_conv, dim=[1, 2]) for log_q_conv in all_log_q]
            all_log_p = [torch.sum(log_p_conv, dim=[1, 2]) for log_p_conv in all_log_p]
            return out_dist, kl_all, all_q, all_p, all_log_q, all_log_p

        return out_dist, kl_all, kl_diag, log_q, log_p


"""
For layer vae, we assume a forward directional dependency.
"""


class LayerVAE(nn.Module):
    def __init__(
        self,
        input_channel=1,
        latent_channel=64,
        layers=[2, 2, 2, 2, 2],
        sampler_dim=40,
        prior_scale=1e-4,
        num_heads=8,
        dim_head=64,
        dropout=0.25,
        sample_decoder=False,
        latent_dim=64,
    ):
        super(LayerVAE, self).__init__()
        num_layers = len(layers)  # How many layers for encoder/decoder tower
        tower_length = num_layers + 1
        self.tower_length = tower_length
        layers = layers + [layers[-1]] * 1

        self.prior_scale = prior_scale  # prior sclae for generating latents: in our experience, 1e-4 is a good choice.
        self.num_layers = num_layers
        self.sample_decoder = sample_decoder  # should we sample the last layer of decoder? True: a learned std, False: a prior std of 1

        self.encoder_tower = nn.ModuleList()
        self.decoder_tower = nn.ModuleList()

        self.enc_combiners = nn.ModuleList()
        self.dec_combiners = nn.ModuleList()

        self.enc_sampler = nn.ModuleList()
        self.dec_sampler = nn.ModuleList()

        # get encoder tower
        """
        Channel:
            layer 1: C -> C
            layer 2: C -> C
            layer 3: C -> C
            layer n > 3: C -> C
        """
        # for n in range(num_layers):
        for n in range(tower_length):
            Cin = latent_channel
            Cout = latent_channel
            self.encoder_tower.append(
                AttnCell(
                    dim=Cin,
                    depth=int(layers[n]),
                    heads=num_heads,
                    dim_head=dim_head,
                    mlp_dim=Cin,
                    out_dim=Cout,
                    dropout=dropout,
                )
            )
            self.enc_combiners.append(EncCombinerCell(Cout, Cout))
            self.enc_sampler.append(
                nn.Sequential(nn.Linear(Cout, sampler_dim, bias=True))
            )

        # get decoder tower
        """
        Channel:
            layer n>3: C -> C
            layer 3: C -> C
            layer 2: C -> C
            layer 1: C -> C
        """
        # for n in range(num_layers):
        for n in range(tower_length):
            Cin = latent_channel
            Cout = latent_channel
            self.decoder_tower.append(
                AttnCell(
                    dim=Cin,
                    depth=int(layers[n]),
                    heads=num_heads,
                    dim_head=dim_head,
                    mlp_dim=Cin,
                    out_dim=Cout,
                    dropout=dropout,
                )
            )
            self.dec_combiners.append(DecCombinerCell(sampler_dim // 2, Cin, Cin))
            # if n<num_layers-1: # As the NVAE, the number of decoder equals the number of encoder minus 1
            if n < tower_length - 1:
                self.dec_sampler.append(
                    nn.Sequential(nn.Linear(Cout, sampler_dim, bias=True))
                )

        self.enc_0 = nn.Sequential(
            nn.ELU(),
            nn.Linear(in_features=latent_channel, out_features=latent_channel),
        )

        """
        For a tensor with shape (b,c,l), we firstly reshape to (b, c, l1, l2)
        """
        ldmp = self.get_maximal_factor_pair(latent_dim)
        lcmp = self.get_maximal_factor_pair(latent_channel)
        self.pre_processor = nn.Sequential(
            nn.Conv1d(
                in_channels=input_channel,
                out_channels=latent_channel,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ),
            Rearrange(
                "b (l1 l2) (l3 l4) -> b (l1 l3) (l2 l4)",
                l1=lcmp[0],
                l2=lcmp[1],
                l3=ldmp[0],
                l4=ldmp[1],
            ),
            nn.Conv1d(
                in_channels=lcmp[0] * ldmp[0],
                out_channels=latent_channel,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ),
            Rearrange("b c l -> b l c", c=latent_channel),
        )
        self.z0_size = [lcmp[1] * ldmp[1], sampler_dim // 2]  # L, 20
        # to generate prior z_1
        prior_ftr0_size = (
            lcmp[1] * ldmp[1],
            latent_channel,
        )  # should align with the siren latents' length
        self.prior_ftr0 = nn.Parameter(
            torch.rand(size=prior_ftr0_size), requires_grad=True
        )
        self.post_processor = nn.Sequential(
            nn.Conv1d(
                lcmp[1] * ldmp[1],
                latent_dim,
                kernel_size=1,
                padding=0,
                stride=1,
                bias=True,
            ),
            # nn.ELU(),
            nn.Linear(latent_channel, input_channel, bias=True),
        )
        if sample_decoder:
            self.dist_embedding = ZCombiner(
                latent_dim, 2 * latent_dim
            )  # should align with the inr latents' length

    def get_maximal_factor_pair(self, number):
        for i in range(1, int(number**0.5) + 1):
            if number % i == 0:
                factor_pair = (i, number // i)
                max_factor_pair = factor_pair
        return max_factor_pair

    def forward(self, z_in, return_meta=False):
        # the z_in should be of shape (b, n, L) = (batch_size, num_layers, latents_length)
        batch_size, num_layers, latent_dim = z_in.size()
        z_in = z_in.reshape(batch_size * num_layers, 1, latent_dim)
        # s1. we want to get posterior zn,...,z2,z1,z0 -> z0,z1,z2,...,zn
        z_n = self.pre_processor(z_in)
        z_n = rearrange(z_n, "(b n) l d -> b n l d", b=batch_size, n=num_layers)
        z_posterior = []
        z_enc_combiners = []
        # for n in range(self.num_layers):
        for n in range(self.tower_length):
            if n < self.num_layers:
                z_enc = self.encoder_tower[n](
                    z_n[:, self.num_layers - n - 1, ...]
                )  # (b, L, 64->64->64->64)
                z_posterior.append(z_enc)
                z_enc_combiners.append(self.enc_combiners[n])
            else:
                z_enc = self.encoder_tower[n](z_enc)
                z_posterior.append(z_enc)
                z_enc_combiners.append(self.enc_combiners[n])
        z_posterior.reverse()
        z_enc_combiners.reverse()

        # s2. get the first posterior
        param_0 = self.enc_0[0](z_posterior[0])  # q_0 (b, L, 64)
        param_0 = self.enc_sampler[self.num_layers - 1](param_0)  # q_0 (b, L, 40)
        mu_q, log_sigma_q = torch.chunk(
            param_0, 2, dim=2
        )  # mu_q, log_sigma_q (b, L,20)
        dist = Normal(mu_q, log_sigma_q)
        z, _ = dist.sample()  # z (b, L, 20)
        log_q = dist.log_p(z)  # (b, L, 20)
        all_q = [dist]
        all_log_q = [log_q]

        # s3. get the first prior
        dist = Normal(
            torch.zeros_like(mu_q),
            torch.zeros_like(log_sigma_q)
            + torch.log(torch.tensor(self.prior_scale, device=mu_q.device)),
        )
        log_p = dist.log_p(z)  # (b, L, 20)
        all_p = [dist]
        all_log_p = [log_p]

        """
        s is a random input sample, It doesn't affect the kld of p and q.
        For further exploration: s can be class label, hierarchy cluster.
        """
        s = self.prior_ftr0.unsqueeze(0)
        batch_size = z.size(0)
        s = s.expand(batch_size, -1, -1)
        s = self.dec_combiners[0](s, z)
        s = self.decoder_tower[0](s)

        # s4. get the combined posterior from the features of generative model
        # for n in range(1,self.num_layers):
        for n in range(1, self.tower_length):
            # a1. get prior
            param = self.dec_sampler[n - 1](s)
            mu_p, log_sigma_p = torch.chunk(param, 2, dim=2)

            # a2. get posterior
            ftr = z_enc_combiners[n](z_posterior[n], s)
            param = self.enc_sampler[self.num_layers - n - 1](ftr)
            mu_q, log_sigma_q = torch.chunk(param, 2, dim=2)
            dist_q = Normal(mu_p + mu_q, log_sigma_p + log_sigma_q)
            # dist_q = Normal(mu_q, log_sigma_q)
            z, _ = dist_q.sample()
            log_q = dist_q.log_p(z)
            all_q.append(dist_q)
            all_log_q.append(log_q)

            dist_p = Normal(mu_p, log_sigma_p)
            log_p = dist_p.log_p(z)
            all_p.append(dist_p)
            all_log_p.append(log_p)

            # a3. combine posterior with prior features
            s = self.dec_combiners[n](s, z)
            s = self.decoder_tower[n](s)
            if n == (self.tower_length - self.num_layers):
                out_tensor = [s]
            if n > (self.tower_length - self.num_layers):
                out_tensor.append(s)

        # the last combined s is sent to post_processor
        out_tensor = torch.stack(out_tensor)
        out_tensor = rearrange(
            out_tensor, "n b l d -> (n b) l d", b=batch_size, n=num_layers
        )
        out_tensor = self.post_processor(out_tensor)
        out_tensor = rearrange(
            out_tensor, "(n b) l d -> b n l d", b=batch_size, n=num_layers
        )

        # Sample decoder or not: True for a learned std, False for a prior std of 1
        if self.sample_decoder:
            z_mu, z_sigma = self.dist_embedding(out_tensor.squeeze(-1))
            out_dist = Normal(z_mu, z_sigma)
        else:
            out_dist = out_tensor

        kl_all = []
        kl_diag = []
        log_p, log_q = 0.0, 0.0
        for q, p, log_q_conv, log_p_conv in zip(all_q, all_p, all_log_q, all_log_p):
            kl_per_var = q.kl(p)

            kl_diag.append(torch.mean(torch.sum(kl_per_var, dim=[1, 2]), dim=0))
            kl_all.append(torch.sum(kl_per_var, dim=[1, 2]))
            log_q += torch.sum(log_q_conv, dim=[1, 2])
            log_p += torch.sum(log_p_conv, dim=[1, 2])
        if return_meta:
            all_log_q = [torch.sum(log_q_conv, dim=[1, 2]) for log_q_conv in all_log_q]
            all_log_p = [torch.sum(log_p_conv, dim=[1, 2]) for log_p_conv in all_log_p]
            return out_dist, kl_all, all_q, all_p, all_log_q, all_log_p

        return out_dist, kl_all, kl_diag, log_q, log_p

    def sample(self, num_samples, t=1.0, fix_z=None):
        z0_size = [num_samples] + self.z0_size
        mu_0 = torch.zeros(z0_size).cuda()
        log_sigma_0 = (
            torch.zeros_like(mu_0) + torch.log(torch.tensor(self.prior_scale)).cuda()
        )
        # log_sigma_0 = torch.zeros_like(mu_0)
        fix_z = [None] * self.tower_length if fix_z is None else fix_z
        if fix_z[0] is not None:
            z = fix_z[0]
        else:
            dist = Normal(mu=mu_0, log_sigma=log_sigma_0, temp=t)
            z, _ = dist.sample()

        zs = [z]

        s = self.prior_ftr0.unsqueeze(0)
        batch_size = z.size(0)
        s = s.expand(batch_size, -1, -1)
        # s = s.cuda()
        # z = z.cuda()
        s = self.dec_combiners[0](s, z)
        s = self.decoder_tower[0](s)

        for n in range(1, self.tower_length):
            # a1. get prior
            if fix_z[n] is not None:
                z = fix_z[n]
            else:
                param = self.dec_sampler[n - 1](s)
                mu_p, log_sigma_p = torch.chunk(param, 2, dim=2)
                dist_p = Normal(mu_p, log_sigma_p, t)
                z, _ = dist_p.sample()
            zs.append(z)

            # a3. combine posterior with prior features
            s = self.dec_combiners[n](s, z)
            s = self.decoder_tower[n](s)
            if n == (self.tower_length - self.num_layers):
                out_tensor = [s]
            if n > (self.tower_length - self.num_layers):
                out_tensor.append(s)

        # the last combined s is sent to post_processor
        out_tensor = torch.stack(out_tensor)
        out_tensor = rearrange(
            out_tensor, "n b l d -> (n b) l d", b=batch_size, n=self.num_layers
        )
        out_tensor = self.post_processor(out_tensor)
        out_tensor = rearrange(
            out_tensor, "(n b) l d -> b n l d", b=batch_size, n=self.num_layers
        )

        # Sample decoder or not: True for a learned std, False for a prior std of 1
        if self.sample_decoder:
            z_mu, z_sigma = self.dist_embedding(out_tensor.squeeze(-1))
            out_dist = Normal(z_mu, z_sigma)
        else:
            out_dist = out_tensor
        return out_dist, zs


if __name__ == "__main__":
    model = LayerVAE().cuda()

    input_sample = torch.randn(2, 5, 64).cuda()
    out_dist, kl_all, kl_diag, log_q, log_p = model(input_sample)
