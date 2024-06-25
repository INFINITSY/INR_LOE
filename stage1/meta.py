import re
import warnings
from collections import OrderedDict

import torch
import torch.nn as nn


class MetaModule(nn.Module):
    """
    Base class for PyTorch meta-learning modules. These modules accept an
    additional argument `params` in their `forward` method.

    Notes
    -----
    Objects inherited from `MetaModule` are fully compatible with PyTorch
    modules from `torch.nn.Module`. The argument `params` is a dictionary of
    tensors, with full support of the computation graph (for differentiation).
    """
    def __init__(self):
        super(MetaModule, self).__init__()
        self._children_modules_parameters_cache = dict()

    def meta_named_parameters(self, prefix='', recurse=True):
        gen = self._named_members(
            lambda module: module._parameters.items()
            if isinstance(module, MetaModule) else [],
            prefix=prefix, recurse=recurse)
        for elem in gen:
            yield elem

    def meta_parameters(self, recurse=True):
        for name, param in self.meta_named_parameters(recurse=recurse):
            yield param

    def get_subdict(self, params, key=None):
        if params is None:
            return None

        all_names = tuple(params.keys())
        if (key, all_names) not in self._children_modules_parameters_cache:
            if key is None:
                self._children_modules_parameters_cache[(key, all_names)] = all_names

            else:
                key_escape = re.escape(key)
                key_re = re.compile(r'^{0}\.(.+)'.format(key_escape))

                self._children_modules_parameters_cache[(key, all_names)] = [
                    key_re.sub(r'\1', k) for k in all_names if key_re.match(k) is not None]

        names = self._children_modules_parameters_cache[(key, all_names)]
        if not names:
            warnings.warn('Module `{0}` has no parameter corresponding to the '
                          'submodule named `{1}` in the dictionary `params` '
                          'provided as an argument to `forward()`. Using the '
                          'default parameters for this submodule. The list of '
                          'the parameters in `params`: [{2}].'.format(
                          self.__class__.__name__, key, ', '.join(all_names)),
                          stacklevel=2)
            return None

        return OrderedDict([(name, params[f'{key}.{name}']) for name in names])
    
class MetaSequential(nn.Sequential, MetaModule):
    __doc__ = nn.Sequential.__doc__

    def forward(self, input, params=None):
        for name, module in self._modules.items():
            if isinstance(module, MetaModule):
                input = module(input, params=self.get_subdict(params, name))
            elif isinstance(module, nn.Module):
                input = module(input)
            else:
                raise TypeError('The module must be either a torch module '
                    '(inheriting from `nn.Module`), or a `MetaModule`. '
                    'Got type: `{0}`'.format(type(module)))
        return input
    
class BatchLinear(nn.Linear, MetaModule):
    '''A linear meta-layer that can deal with batched weight matrices and biases, as for instance output by a
    hypernetwork.'''
    __doc__ = nn.Linear.__doc__

    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())

        bias = params.get('bias', None)
        weight = params['weight']

        output = input.matmul(weight.permute(*[i for i in range(len(weight.shape) - 2)], -1, -2))
        output += bias.unsqueeze(-2)
        return output