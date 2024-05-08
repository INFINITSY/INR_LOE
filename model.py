import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import matplotlib.pyplot as plt
import torch_geometric
from datasets import CIFAR10Dataset, LatticeDataset
from meta import MetaModule, MetaSequential, BatchLinear

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

class ConditionalGateModule(nn.Module):
    def __init__(self, latent_size, num_exps=[64, 64, 64, 64]):
        super().__init__()
        self.num_exps = num_exps
        self.nets = nn.ModuleList()
        # output gating vector for each layer plus mean and std for subsequent layer
        for i in range(len(num_exps)-1):
            self.nets.append(nn.Linear(latent_size, num_exps[i] + 2 * latent_size))
        self.nets.append(nn.Linear(latent_size, num_exps[-1]))

        # init output of each layer to be uniform
        for i, net in enumerate(self.nets):
            if i < len(self.nets) - 1:
                net.bias.data[:num_exps[i]].fill_(1/num_exps[i])
                # init mean and log_var to be 0
                net.bias.data[num_exps[i]:].fill_(0)
            else:
                net.bias.data.fill_(1/num_exps[-1])

    def forward(self, latents):
        # latents is N_imgs x N_layers x latent_size
        latent_size = latents.shape[-1]
        gates = []
        means = []
        log_vars = []
        mean = torch.zeros_like(latents[:, 0]) # N_imgs x latent_size
        log_var = torch.zeros_like(latents[:, 0]) # N_imgs x latent_size

        for i, net in enumerate(self.nets):
            # reparametrize the latents
            std = torch.exp(0.5 * log_var)
            latents_rprm = mean + std * latents[:, i] # N_imgs x latent_size
            gate_raw = net(latents_rprm) # N_imgs x (num_exps[i] + 2 * latent_size)

            if i < len(self.nets) - 1:
                gates.append(gate_raw[:, :self.num_exps[i]]) # N_imgs x num_exps[i]
                # for the next layer
                mean = gate_raw[:, self.num_exps[i]:-latent_size] # N_imgs x latent_size
                log_var = gate_raw[:, -latent_size:] # N_imgs x latent_size
                # record mean and log_var
                means.append(mean)
                log_vars.append(log_var)
            else:
                gates.append(gate_raw) # N_imgs x num_exps[-1]

        return torch.cat(gates, dim=1), means, log_vars
        
class SeparateGateModule(nn.Module):
    def __init__(self, latent_size, num_exps=[64, 64, 64, 64]):
        super().__init__()
        self.num_exps = num_exps
        self.nets = nn.ModuleList()
        for i in range(len(num_exps)):
            self.nets.append(nn.Linear(latent_size, num_exps[i]))

        # init output of each layer to be uniform
        for i, net in enumerate(self.nets):
            net.bias.data.fill_(1/num_exps[i])

    def forward(self, latents):
        # latents is N_imgs x N_layers x latent_size
        gates = []
        for i, net in enumerate(self.nets):
            gate = net(latents[:, i]) # N_imgs x num_exps[i]
            gates.append(gate) 

        return torch.cat(gates, dim=1) # N_imgs x sum(num_exps)
    

class INRLoe(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=256, output_dim=3, 
                 num_hidden=4,
                 num_exps=[8, 16, 64, 256, 1024], 
                 ks = [4, 4, 32, 32, 256],
                 latent_size=64, gate_type='separate',
                 noisy_gating=False, noise_module=None,
                 ):
        super(INRLoe, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_hidden = num_hidden
        self.net_param = []
        self.net = []
        self.nl = Sine()
        self.num_exps = num_exps
        self.ks = ks
        self.noisy_gating = noisy_gating
        self.gate_type = gate_type

        if self.noisy_gating and noise_module is not None:
            self.noise_generator = noise_module(output_size=sum(self.num_exps))
            self.softplus = nn.Softplus()

        # for param
        self.net_param.append(nn.Sequential(
            nn.Linear(input_dim, hidden_dim * self.num_exps[0]), self.nl
            ))
        for i in range(num_hidden-1):
            self.net_param.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * self.num_exps[i+1]), self.nl
            ))
        self.net_param.append(nn.Linear(hidden_dim, output_dim * self.num_exps[-1]))
        self.net_param = nn.Sequential(*self.net_param)
        # self.net.apply(init_weights_relu)
        self.net_param.apply(sine_init)
        self.net_param[0].apply(first_layer_sine_init)

        # print net weight shape
        for name, weights_all in self.net_param.named_parameters():
            print(f'{name}: {weights_all.shape}')

        # for inference
        self.net.append(MetaSequential(
            BatchLinear(input_dim, hidden_dim), self.nl
            ))
        for i in range(num_hidden-1):
            self.net.append(MetaSequential(
                BatchLinear(hidden_dim, hidden_dim), self.nl
            ))
        self.net.append(BatchLinear(hidden_dim, output_dim))
        self.net = MetaSequential(*self.net)
        self.net.apply(sine_init)
        self.net[0].apply(first_layer_sine_init)
        for name, weights_all in self.net.named_parameters():
            print(f'{name}: {weights_all.shape}')

        output_size = sum(self.num_exps)

        if self.gate_type == 'conditional':
            self.gate_module = ConditionalGateModule(latent_size, num_exps=self.num_exps)
        elif self.gate_type == 'separate':
            self.gate_module = SeparateGateModule(latent_size, num_exps=self.num_exps)
        elif self.gate_type == 'shared':
            self.gate_module = nn.Linear(latent_size, output_size)
            for i, num_exp in enumerate(self.num_exps):
                self.gate_module.bias.data[i*num_exp:(i+1)*num_exp].fill_(1/num_exp)
        else:
            raise ValueError(f'Unsupported gate type: {self.gate_type}')

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

        return gates #, bias, load
    
    def get_combined_weight(self, gates):
        # gates is a list of len(num_exps) x N_imgs x num_exps[i]
        # return combined weight of shape N_imgs x (sum(num_exps))

        params = dict()
        for i, (name, weights_all) in enumerate(self.net_param.named_parameters()):
            w_size = weights_all.size()
            l = i // 2 # layer index
            gate = gates[l] # N_imgs x N_exps
            N_imgs, N_exps = gate.shape
            weights_all = weights_all.view(N_exps, -1)
            combined_weight = torch.matmul(gate, weights_all) # N_imgs x (hidden_dim * input_dim)
            combined_weight = combined_weight.view(
                [N_imgs, w_size[0]//N_exps] + list(w_size[1:])) # N_imgs x hidden_dim x input_dim
            params[name] = combined_weight

        return params

    def forward(self, latents, coords, top_k=False, blend_alphas=[0, 0, 0, 0, 0]):

        if self.gate_type == 'conditional':
            raw_gates, means, log_vars = self.gate_module(latents) # N_imgs x sum(num_exps)
            mu_var = {'means': means, 'log_vars': log_vars}
        else:
            raw_gates = self.gate_module(latents)
            mu_var = None

        # split gates to according to self.num_exps
        gates = torch.split(raw_gates, self.num_exps, dim=1) # len(num_exps) x N_imgs x num_exps[i]

        # to list
        gates = list(gates)
        
        if top_k:
            gates = self.noisy_top_k_gating(latents, gates)

        # blend the gates with uniform weights
        for i, alpha in enumerate(blend_alphas):
            gates[i] = alpha / self.num_exps[i] + (1 - alpha) * gates[i]

        importance = [torch.sum(gate, dim=0) for gate in gates] # len(num_exps) x num_exps[i]

        params = self.get_combined_weight(gates) # dict of combined weights
        
        x = coords
        x = self.net(x, params=params)
        
        x = torch.sigmoid(x)

        return x, gates, importance, mu_var
    
    def get_parameters(self):
        # return both the parameters of the network and the gate module
        return list(self.net_param.parameters()) + list(self.gate_module.parameters())
    

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim, condition_dim=None):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.condition = condition_dim is not None

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim) if not self.condition else nn.Linear(input_dim + condition_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.1) ,
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.1) ,
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.1) ,
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.1) ,
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.1) 
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim) if not self.condition else nn.Linear(latent_dim + condition_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.1) ,
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.1) ,
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.1) ,
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.1) ,
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.1) ,
            nn.Linear(hidden_dim, input_dim),
            # nn.Sigmoid()  # Use sigmoid if the data is normalized between 0 and 1
        )
    
    def encode(self, x, c=None):
        if self.condition:
            x = torch.cat([x, c], dim=1)
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_var(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def decode(self, z, c=None):
        if self.condition:
            z = torch.cat([z, c], dim=1)
        return self.decoder(z)
    
    def forward(self, x, c=None):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar
    
    
class CVAE(nn.Module):
    def __init__(self, input_dim, condition_dim, latent_dim, hidden_dim):
        super(CVAE, self).__init__()
        self.input_dim = input_dim
        self.condition_dim = condition_dim
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + condition_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + condition_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()  # Use sigmoid if the data is normalized between 0 and 1
        )
    
    def encode(self, x, c):
        combined = torch.cat([x, c], dim=1)
        h = self.encoder(combined)
        return self.fc_mu(h), self.fc_var(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def decode(self, z, c):
        combined = torch.cat([z, c], dim=1)
        return self.decoder(combined)
    
    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar
    

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