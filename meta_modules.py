"""
meta_modules.py
---------------
Base classes for MAML-compatible (inner-loop adaptable) modules.

Provides:
    MetaModule   — nn.Module subclass that exposes meta_named_parameters /
                   meta_parameters for inner-loop gradient updates.
    MetaLinear   — nn.Linear subclass that accepts an external `params`
                   dict so adapted weights can be injected at forward time.
    get_subdict  — Helper to extract a sub-namespace from a flat params dict.
"""

import math
import re
from collections import OrderedDict

import torch.nn as nn
import torch.nn.functional as F


class MetaModule(nn.Module):
    """nn.Module that exposes only MetaModule-registered parameters.

    Only parameters belonging to MetaModule submodules are yielded by
    meta_named_parameters / meta_parameters.  This lets the inner loop
    update a selective subset of the network (e.g. projection heads only)
    while leaving frozen backbones untouched.
    """

    def meta_named_parameters(self, prefix: str = '', recurse: bool = True):
        gen = self._named_members(
            lambda module: module._parameters.items()
            if isinstance(module, MetaModule) else [],
            prefix=prefix,
            recurse=recurse,
        )
        yield from gen

    def meta_parameters(self, recurse: bool = True):
        for _, param in self.meta_named_parameters(recurse=recurse):
            yield param


class MetaLinear(nn.Linear, MetaModule):
    """Linear layer whose weights can be overridden at forward time.

    When `params` is supplied the layer uses the tensors stored in that
    dict (keyed by 'weight' and 'bias') instead of its own registered
    parameters.  This is the mechanism that lets phi' flow through the
    classification head without in-place parameter mutation.
    """

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())
        weight = params.get('weight', self.weight)
        bias   = params.get('bias',   self.bias)
        return F.linear(input, weight, bias)


def get_subdict(dictionary: OrderedDict, key: str = None) -> OrderedDict:
    """Return the sub-namespace of *dictionary* matching the given key prefix.

    Example:
        d = {'fc.weight': w, 'fc.bias': b, 'res_linear.weight': w2}
        get_subdict(d, 'fc')  ->  {'weight': w, 'bias': b}

    Args:
        dictionary : flat OrderedDict of named parameters.
        key        : dot-separated prefix to extract (None returns full dict).

    Returns:
        OrderedDict with the prefix stripped from matching keys.
    """
    if dictionary is None:
        return None
    if key is None or key == '':
        return dictionary
    key_re = re.compile(r'^{0}\.(.+)'.format(re.escape(key)))
    return OrderedDict(
        (key_re.sub(r'\1', k), v)
        for k, v in dictionary.items()
        if key_re.match(k) is not None
    )
