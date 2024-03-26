import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import matplotlib.pyplot as plt
import torch_geometric
from datasets import CIFAR10Dataset, LatticeDataset

### Taken from official SIREN repo
class Sine(nn.Module):
    def __init(self):
        super().__init__()

    def forward(self, input):
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(30 * input)
    

class PositionalEncoding(nn.Module):
    '''Module to add positional encoding as in NeRF [Mildenhall et al. 2020].'''
    def __init__(self, in_features, num_frequencies=-1, sidelength=None, use_nyquist=True):
        super().__init__()

        self.in_features = in_features

        self.num_frequencies = num_frequencies
        if self.num_frequencies < 0:
            if self.in_features == 3:
                self.num_frequencies = 10
            elif self.in_features == 2:
                assert sidelength is not None
                if isinstance(sidelength, int):
                    sidelength = (sidelength, sidelength)
                self.num_frequencies = 4
                if use_nyquist:
                    self.num_frequencies = self.get_num_frequencies_nyquist(min(sidelength[0], sidelength[1]))
            elif self.in_features == 1:
                assert sidelength is not None
                self.num_frequencies = 4
                if use_nyquist:
                    self.num_frequencies = self.get_num_frequencies_nyquist(sidelength)

    @property
    def out_dim(self):
        return self.in_features + 2 * self.in_features * self.num_frequencies

    @property
    def flops(self):
        return self.in_features + (2 * self.in_features * self.num_frequencies) * 2

    def get_num_frequencies_nyquist(self, samples):
        nyquist_rate = 1 / (2 * (2 * 1 / samples))
        return int(math.floor(math.log(nyquist_rate, 2)))

    def forward(self, coords):
        coords = coords.view(coords.shape[0], -1, self.in_features)

        coords_pos_enc = coords
        for i in range(self.num_frequencies):
            for j in range(self.in_features):
                c = coords[..., j]

                sin = torch.unsqueeze(torch.sin((2 ** i) * np.pi * c), -1)
                cos = torch.unsqueeze(torch.cos((2 ** i) * np.pi * c), -1)

                coords_pos_enc = torch.cat((coords_pos_enc, sin, cos), axis=-1)

        return coords_pos_enc.reshape(coords.shape[0], -1, self.out_dim)
    

class CodebookImgEncoder(nn.Module):
    
    def __init__(self, num_images, output_size, max_norm=None, norm_type=2.):
        super().__init__()
        self.codebook = nn.Embedding(num_images, output_size, max_norm=max_norm, norm_type=norm_type)

    @property
    def flops(self):
        return 0

    def forward(self, sample_ids):
        return self.codebook(sample_ids) # [N_samples, out_dim]
    

class SimpleConvImgEncoder(nn.Module):

    def __init__(self, input_size=3, hidden_dim=256, num_layers=3, output_size=124):
        super().__init__()
        convs = [nn.Conv2d(input_size, hidden_dim, 5, 1, 2), nn.ReLU()]
        for i in range(num_layers-1):
            convs.append(nn.Conv2d(hidden_dim, hidden_dim, 5, 1, 2))
            convs.append(nn.ReLU())
        self.convs = nn.Sequential(*convs)
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        # x = x.permute(0, 3, 1, 2)
        x = self.convs(x)
        B, C = x.shape[0], x.shape[1]
        x = torch.sum(x.reshape(B, C, -1), -1)
        x = self.fc(x)
        return x


class Conv2dResBlock(nn.Module):
    '''Aadapted from https://github.com/makora9143/pytorch-convcnp/blob/master/convcnp/modules/resblock.py'''
    def __init__(self, in_channel, out_channel=128, downsample=False):
        super().__init__()
        self.downsample = downsample
        stride = 2 if downsample else 1
        self.convs = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 5, stride, 2),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, 5, 1, 2),
            nn.ReLU()
        )

        self.downsample_layer = nn.Conv2d(in_channel, out_channel, 1, 2, 0) if downsample else None
        self.final_relu = nn.ReLU()

    def forward(self, x):
        shortcut = x if not self.downsample else self.downsample_layer(x)
        output = self.convs(x)
        output = self.final_relu(output + shortcut)
        return output
    

class ConvImgEncoder(nn.Module):
    def __init__(self, channel, image_resolution):
        super().__init__()

        # conv_theta is input convolution
        self.conv_theta = nn.Conv2d(channel, 128, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)

        self.cnn = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(),
            Conv2dResBlock(256, 256),
            Conv2dResBlock(256, 256),
            Conv2dResBlock(256, 256),
            Conv2dResBlock(256, 256),
            nn.Conv2d(256, 256, 1, 1, 0)
        )

        self.relu_2 = nn.ReLU(inplace=True)
        self.fc = nn.Linear(image_resolution*image_resolution, 1)


    def forward(self, I):
        o = self.relu(self.conv_theta(I))
        o = self.cnn(o)

        o = self.fc(self.relu_2(o).view(o.shape[0], 256, -1)).squeeze(-1)
        return o
    

class DownConvImgEncoder(nn.Module):
    def __init__(self, in_channel, out_channel, image_resolution):
        super().__init__()

        # conv_theta is input convolution
        self.conv_theta = nn.Conv2d(in_channel, 64, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)

        self.cnn = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.ReLU(),
            Conv2dResBlock(128, 128),
            Conv2dResBlock(128, 256, downsample=True),
            Conv2dResBlock(256, 256),
            Conv2dResBlock(256, 512, downsample=True),
            nn.Conv2d(512, 512, 1, 1, 0)
        )

        self.relu_2 = nn.ReLU(inplace=True)
        out_res = image_resolution // 8
        self.fc = nn.Linear(512*out_res*out_res, out_channel)


    def forward(self, I):
        o = self.relu(self.conv_theta(I))
        o = self.relu_2(self.cnn(o))

        o = o.view(o.shape[0], -1) # flatten
        o = self.fc(o) # [B, 512]
        # o = self.fc(self.relu_2(o).view(o.shape[0], 256, -1)).squeeze(-1)
        return o
    

class ResConvImgEncoder(nn.Module):

    def __init__(self, input_size, output_size, image_resolution):
        super().__init__()
        self.convs = ConvImgEncoder(input_size, image_resolution)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(256, output_size)

    def forward(self, x):
        # x = model_input['imgs'].permute(0, 3, 1, 2) # [B, C, H, W]
        x = self.convs(x) # [B, 256]
        x = self.relu(x) # [B, 256]
        x = self.fc(x) # [B, out_dim]
        return x


class ResDownConvImgEncoder(nn.Module):

    def __init__(self, input_size, latent_size, output_size, image_resolution):
        super().__init__()
        self.convs = DownConvImgEncoder(input_size, latent_size, image_resolution)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(latent_size, output_size)

    def forward(self, x):
        # x = model_input['imgs'].permute(0, 3, 1, 2) # [B, C, H, W]
        x = self.convs(x) # [B, 256]
        x = self.relu(x) # [B, 256]
        x = self.fc(x) # [B, out_dim]
        return x
    

class ResNet18ImgEncoder(nn.Module):

    def __init__(self, output_size):
        super().__init__()
        self.convs = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=True)
        self.convs.fc = nn.Linear(512, output_size)

    def forward(self, x):
        x = self.convs(x)
        return x
    

class MoECombiner(torch_geometric.nn.conv.MessagePassing):
    
    def __init__(self):
        super().__init__(aggr='add')  # Use 'add' aggregation.

    def forward(self, expert_outputs, gates):
        # Determine the structure of expert_outputs
        if expert_outputs.dim() == 2:  # Shared expert outputs case: 8xD
            return self.process_shared_experts(expert_outputs, gates)
        elif expert_outputs.dim() == 3:  # Unique expert outputs per image: 64x8xD
            return self.process_unique_experts(expert_outputs, gates)
        else:
            raise ValueError("Unsupported expert_outputs shape")

    def process_shared_experts(self, expert_outputs, gates):
        expert_indices = torch.nonzero(gates, as_tuple=True)
        edge_index = torch.stack([expert_indices[1], expert_indices[0]], dim=0)
        edge_weights = gates[expert_indices[0], expert_indices[1]].unsqueeze(-1)
        num_experts, num_images = expert_outputs.shape[0], gates.shape[0]

        # Here, expert_outputs are shared across all images
        out = self.propagate(edge_index, x=(expert_outputs, None), edge_weights=edge_weights, size=(num_experts, num_images))
        return out

    def process_unique_experts(self, expert_outputs, gates):
        num_images, num_experts, D = expert_outputs.shape
        
        expert_outputs_flat = expert_outputs.reshape(-1, D)
        gates_flat = gates.view(-1)
        nonzero_indices = gates_flat.nonzero().squeeze()

        images_indices = torch.arange(num_images).repeat_interleave(num_experts).to(gates.device)
        experts_indices = torch.tile(torch.arange(num_experts), (num_images,)).to(gates.device)
        all_edges = torch.stack([images_indices, experts_indices], dim=0)
        
        edge_index = all_edges[:, nonzero_indices]
        edge_weights = gates_flat[nonzero_indices].unsqueeze(-1)
        
        out = self.propagate(edge_index, x=(expert_outputs_flat, None), edge_weights=edge_weights, size=(num_images*num_experts, num_images))
        return out

    def message(self, x_j, x_i, edge_weights):
        return x_j * edge_weights

class INRLoe(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=256, output_dim=3, 
                 num_hidden=4, image_resolution=32,
                 num_exps=[8, 16, 64, 256, 1024], 
                 ks = [4, 4, 32, 32, 256],
                 noisy_gating=False, noise_module=None,
                 merge_before_act=False, bias=False):
        super(INRLoe, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_hidden = num_hidden
        self.net = nn.ModuleList()
        self.nl = Sine()
        self.num_exps = num_exps
        self.ks = ks
        self.noisy_gating = noisy_gating
        self.merge_before_act = merge_before_act
        self.bias = bias
        # self.top_k = top_k

        if self.noisy_gating and noise_module is not None:
            self.noise_generator = noise_module(output_size=sum(self.num_exps))
            self.softplus = nn.Softplus()

        # self.map = PositionalEncoding(in_features=2,
        #         num_frequencies=10,
        #         sidelength=32,
        #         use_nyquist=True
        #     )
        # input_dim = self.map.out_dim

        self.net.append(
            nn.Linear(input_dim, hidden_dim * self.num_exps[0])
            )
        for i in range(num_hidden-1):
            self.net.append(
                nn.Linear(hidden_dim, hidden_dim * self.num_exps[i+1])
            )
        self.net.append(nn.Linear(hidden_dim, output_dim * self.num_exps[-1]))

        output_size = sum(self.num_exps) + hidden_dim * num_hidden + output_dim if self.bias \
            else sum(self.num_exps)
        # self.gate_module = ResDownConvImgEncoder(input_size=3, latent_size=512,
        #                                          output_size=output_size, 
        #                                          image_resolution=image_resolution)

        self.gate_module = ResNet18ImgEncoder(output_size=output_size)
        
        # self.gate_module = ResConvImgEncoder(input_size=3, output_size=sum(self.num_exps),
        #                                      image_resolution=image_resolution)
        # self.gate_module = CodebookImgEncoder(num_images=5000, output_size=sum(self.num_exps))

        self.combiner = MoECombiner()

        # self.net.apply(init_weights_relu)
        self.net.apply(sine_init)

        self.net[0].apply(first_layer_sine_init)

    def noisy_top_k_gating(self, x, raw_gates, noise_epsilon=1e-2):
        """Noisy top-k gating.
          See paper: https://arxiv.org/abs/1701.06538.
          Args:
            x: input Tensor with shape [batch_size, input_size]
            noise_epsilon: a float
          Returns:
            gates: a Tensor with shape [batch_size, num_experts]
            load: a Tensor with shape [num_experts]
        """
        # bias = 0.
        # if self.bias:
        #     clean_logits = raw_logits[:, :-self.output_size] # [bs, num_exps]
        #     bias = raw_logits[:, -self.output_size:] # [bs, dim]
        # else:
        clean_gates = raw_gates # len(num_exps) x N_imgs x num_exps[i]

        if self.noisy_gating and self.training:
            # raw_noise_stddev = x @ self.w_noise
            raw_noise_stddev = self.noise_generator(x)
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
            noise_stddev = torch.split(noise_stddev, self.num_exps, dim=1)
            noisy_gates = [clean_gate + torch.randn_like(clean_gate) * noise 
                           for clean_gate, noise in zip(clean_gates, noise_stddev)]
            gates = noisy_gates
        else:
            gates = clean_gates

        # calculate topk + 1 that will be needed for the noisy gates
        for i in range(len(gates)):
            gate = gates[i]
            k = self.ks[i]
            num_exp = self.num_exps[i]
            top_logits, top_indices = torch.abs(gate).topk(min(k + 1, num_exp), dim=1)
            top_k_logits = top_logits[:, :k]
            top_k_indices = top_indices[:, :k]
            # top_k_gates = self.softmax(top_k_logits)
            top_k_gates = torch.gather(gate, 1, top_k_indices)

            zeros = torch.zeros_like(gate, requires_grad=True).to(gate.device)
            gates[i] = zeros.scatter(1, top_k_indices, top_k_gates) # clear entries out of activated gates

        # if self.noisy_gating and self.k < self.num_experts and self.training:
        #     load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        # else:
        #     load = self._gates_to_load(gates)
        return gates #, bias, load
        
    def forward(self, img, coords, top_k=False, softmax=False, 
                noise_gates=[0, 0, 0, 0, 0], blend_alphas=[0, 0, 0, 0, 0]):

        raw_gates = self.gate_module(img) # N_imgs x sum(num_exps)
        # split gates to according to self.num_exps
        if self.bias:
            gates = raw_gates[:, :sum(self.num_exps)] # N_imgs x sum(num_exps)
            gates = torch.split(gates, self.num_exps, dim=1) # len(num_exps) x N_imgs x num_exps[i]

            bias = raw_gates[:, sum(self.num_exps):]
            dims = [self.hidden_dim] * self.num_hidden + [self.output_dim]
            bias = torch.split(bias, dims, dim=1) # len(num_exps) x N_imgs x hidden_dim[i]
        else:
            gates = torch.split(raw_gates, self.num_exps, dim=1) # len(num_exps) x N_imgs x num_exps[i]

        # to list
        gates = list(gates)
        # add noise to gates
        if not self.training:
            for gate, noise in zip(gates, noise_gates):
                gate += noise * torch.randn_like(gate)
        
        if top_k:
            gates = self.noisy_top_k_gating(img, gates)
        elif softmax:
            # apply softmax to each gate
            gates = [F.softmax(gate, dim=1) for gate in gates] # len(num_exps) x N_imgs x num_exps[i]
        else:
            # l2 normalize each gate
            gates = [F.normalize(gate, p=2, dim=1) for gate in gates] # len(num_exps) x N_imgs x num_exps[i]
        
        # blend the gates with uniform weights
        for i, alpha in enumerate(blend_alphas):
            gates[i] = alpha / self.num_exps[i] + (1 - alpha) * gates[i]

        importance = [torch.sum(gate, dim=0) for gate in gates] # len(num_exps) x num_exps[i]

        N_imgs = img.shape[0]
        N_coords = coords.shape[0]
        
        # x = self.map(coords) # N_coords x in_dim
        x = coords
        for i, (gate, layer) in enumerate(zip(gates, self.net)):
            x = layer(x) # N_coords x (hidden_dim * num_exps[i]) for the first layer, 
            # N_imgs x N_coords x (hidden_dim * num_exps[i]) for the rest 
            if i < len(self.net) - 1 and not self.merge_before_act:
                x = self.nl(x)
            N_exp = self.num_exps[i]
            if i == 0:
                x = x.reshape(N_coords, -1, N_exp) # N_coords x hidden_dim x num_exps[i]
                x = x.permute(2, 0, 1) # num_exps[i] x N_coords x hidden_dim
                x = x.reshape(N_exp, -1) # num_exps[i] x (N_coords * hidden_dim)
                # shape of gate is N_imgs x num_exps[i]
                if top_k:
                    x = self.combiner(x, gate) # N_imgs x (N_coords * hidden_dim)
                else:
                    x = torch.matmul(gate, x) # N_imgs x (N_coords * hidden_dim)
            else:
                x = x.reshape(N_imgs, N_coords, -1, N_exp) # N_imgs x N_coords x hidden_dim x num_exps[i]
                x = x.permute(0, 3, 1, 2) # N_imgs x num_exps[i] x N_coords x hidden_dim
                x = x.reshape(N_imgs, N_exp, -1) # N_imgs x num_exps[i] x (N_coords * hidden_dim)
                if top_k:
                    # x = self.combiner(x, gate) 
                    x = torch.bmm(gate.unsqueeze(1), x).squeeze(1) # N_imgs x (N_coords * hidden_dim)
                else:
                    x = torch.bmm(gate.unsqueeze(1), x).squeeze(1) # N_imgs x (N_coords * hidden_dim)
            x = x.reshape(N_imgs, N_coords, -1) # N_imgs x N_coords x hidden_dim
            if self.bias:
                x += bias[i].unsqueeze(1) # N_imgs x 1 x hidden_dim
            # apply non-linearity except for the last layer
            if i < len(self.net) - 1 and self.merge_before_act:
                x = self.nl(x)

        return x, gates, importance


def init_weights_relu(m):
    if type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            stdv = 1. / math.sqrt(m.weight.size(1))
            m.weight.data.uniform_(-stdv, stdv)
            if m.bias is not None:
                m.bias.data.uniform_(-stdv, stdv)


def sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)


def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-1 / num_input, 1 / num_input)