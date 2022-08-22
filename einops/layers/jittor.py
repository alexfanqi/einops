from typing import Optional, Dict

import jittor

from . import RearrangeMixin, ReduceMixin
from ._einmix import _EinmixMixin

__author__ = 'Alex Fan'


class Rearrange(RearrangeMixin, jittor.nn.Module):
    def execute(self, input):
        return self._apply_recipe(input)


class Reduce(ReduceMixin, jittor.nn.Module):
    def execute(self, input):
        return self._apply_recipe(input)


class EinMix(_EinmixMixin, jittor.nn.Module):
    def _create_parameters(self, weight_shape, weight_bound, bias_shape, bias_bound):
        self.weight = jittor.zeros(weight_shape).uniform_(-weight_bound, weight_bound)
        if bias_shape is not None:
            self.bias = jittor.zeros(bias_shape).uniform_(-bias_bound, bias_bound)
        else:
            self.bias = None

    def _create_rearrange_layers(self,
                                 pre_reshape_pattern: Optional[str],
                                 pre_reshape_lengths: Optional[Dict],
                                 post_reshape_pattern: Optional[str],
                                 post_reshape_lengths: Optional[Dict],
                                 ):
        self.pre_rearrange = None
        if pre_reshape_pattern is not None:
            self.pre_rearrange = Rearrange(pre_reshape_pattern, **pre_reshape_lengths)

        self.post_rearrange = None
        if post_reshape_pattern is not None:
            self.post_rearrange = Rearrange(post_reshape_pattern, **post_reshape_lengths)

    def execute(self, input):
        if self.pre_rearrange is not None:
            input = self.pre_rearrange(input)
        result = jittor.linalg.einsum(self.einsum_pattern, input, self.weight)
        if self.bias is not None:
            result += self.bias
        if self.post_rearrange is not None:
            result = self.post_rearrange(result)
        return result
