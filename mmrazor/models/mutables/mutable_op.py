# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from mmrazor.models.builder import build_op
from mmrazor.utils import master_only_print
from .mixins import DiffentiableMixin, DynamicMixin, OneShotMixin


class OneShotOP(nn.Module, OneShotMixin):

    def __init__(self) -> None:
        super().__init__()
        self.candidate_ops = nn.ModuleDict()

    def init_ops(self, ops):
        for name, op_cfg in ops.items():
            assert name not in self.candidate_ops
            self.candidate_ops[name] = build_op(op_cfg)

    @property
    def choices(self):
        return self.candidate_ops.keys()

    def deploy(self, chosen):

        for name in self.choices:
            if name != chosen:
                self.candidate_ops.pop(name)
        self.deployed = True

    def forward(self, x, sampled=None):
        if self.deployed:
            assert sampled is None
            return self.forward_deploy(x)
        elif sampled:
            return self.forward_sample(x, sampled)

    def forward_deploy(self, x):
        assert self.num_choices == 1
        chosen = self.choices[0]
        return self.candidate_ops[chosen](x)

    def forward_sample(self, x, sampled):
        return self.candidate_ops[sampled](x)


class DifferentiableOP(nn.Module, DiffentiableMixin):

    def __init__(self) -> None:
        super().__init__()
        self.candidate_ops = nn.ModuleDict()

    def init_ops(self, ops):
        for name, op_cfg in ops.items():
            assert name not in self.candidate_ops
            self.candidate_ops[name] = build_op(op_cfg)

    @property
    def choices(self):
        return self.candidate_ops.keys()

    @property
    def num_choices(self):
        return len(self.candidate_ops.keys())

    def deploy(self, chosen):

        for name in self.choices:
            if name != chosen:
                self.candidate_ops.pop(name)
        self.deployed = True

    def forward(self, x, arch_params=None):
        if self.deployed:
            assert arch_params is None
            return self.forward_deploy(x)
        elif arch_params:
            return self.forward_train(x, arch_params)

    def forward_deploy(self, x):
        assert self.num_choices == 1
        chosen = self.choices[0]
        return self.candidate_ops[chosen](x)

    def forward_train(self, x, arch_params):

        probs = F.softmax(arch_params, dim=-1)
        outputs = [op(x) for op in self.candidate_ops.values()]
        outputs = [o * p for o, p in zip(outputs, probs)]

        return sum(outputs)


class DynamicKernelConv2d(DynamicMixin, nn.Conv2d):

    def __init__(self,
                 in_channels,
                 out_channels,
                 dynamic_kernel_size,
                 stride=1,
                 padding_mode='same',
                 dilation=1,
                 groups=1,
                 bias=True):
        super().__init__(in_channels, out_channels, max(dynamic_kernel_size),
                         stride, 0, dilation, groups, bias)

        self.dynamic_kernel_size = sorted(
            list(dynamic_kernel_size), reverse=True)
        self._max_kernel_size = max(self.dynamic_kernel_size)

        for i in range(len(self.dynamic_kernel_size) - 1):
            source_kernel_size = self.dynamic_kernel_size[i]
            target_kernel_size = self.dynamic_kernel_size[i + 1]
            transform_matrix_name = self._get_transform_matrix_name(
                src=source_kernel_size, tar=target_kernel_size)
            self.register_parameter(
                name=transform_matrix_name,
                param=nn.Parameter(torch.eye(target_kernel_size**2)))

        self._deployed = False
        self._choice_probs = [1 / len(self.dynamic_kernel_size)] * len(
            self.dynamic_kernel_size)

    @property
    def num_choices(self) -> int:
        return len(self.dynamic_kernel_size)

    @property
    def min_choice(self):
        return self.choices[-1]

    @property
    def max_choice(self):
        return self.choices[0]

    @property
    def choice_probs(self):
        assert sum(self._choice_probs) == 1
        return self._choice_probs

    @choice_probs.setter
    def choice_probs(self, value):
        self._choice_probs = value

    @property
    def deployed(self):
        return self._deployed

    @deployed.setter
    def deployed(self, value):
        self._deployed = value

    @property
    def choices(self):
        # TODO verify sorted
        return self.dynamic_kernel_size

    def deploy(self, chosen):
        weight = self._get_weight(chosen)
        padding = self._get_padding(chosen)
        self.weight = nn.Parameter(weight)
        self.padding = (padding, padding)
        self.deployed = True

    def forward_deploy(self, x):
        # import pdb;pdb.set_trace()
        return nn.Conv2d.forward(self, x)

    def forward_sample(self, input: Tensor, sampled=None) -> Tensor:
        weight = self._get_weight(sampled)
        padding = self._get_padding(sampled)

        return F.conv2d(
            input=input,
            weight=weight,
            bias=self.bias,
            stride=self.stride,
            padding=padding,
            dilation=self.dilation,
            groups=self.groups)

    @staticmethod
    def _get_transform_matrix_name(src: int, tar: int) -> str:
        return f'transform_matrix_{src}to{tar}'

    def _get_weight(self, kernel_size) -> torch.Tensor:
        if kernel_size == self._max_kernel_size:
            return self.weight

        current_weight = self.weight[:, :, :, :]
        for i in range(len(self.choices) - 1):
            source_kernel_size = self.choices[i]
            if source_kernel_size <= kernel_size:
                break
            target_kernel_size = self.choices[i + 1]
            transform_matrix = getattr(
                self,
                self._get_transform_matrix_name(
                    src=source_kernel_size, tar=target_kernel_size))
            master_only_print(f'source_kernel_size: {source_kernel_size}, '
                              f'target_kernel_size: {target_kernel_size}')
            master_only_print(f'transform matrix: {transform_matrix.shape}')

            start_offset, end_offset = self._get_current_kernel_pos(
                source_kernel_size=source_kernel_size,
                target_kernel_size=target_kernel_size)
            target_weight = current_weight[:, :, start_offset:end_offset,
                                           start_offset:end_offset]
            target_weight = target_weight.reshape(-1, target_kernel_size**2)
            target_weight = F.linear(target_weight, transform_matrix)
            target_weight = target_weight.reshape(
                self.weight.size(0), self.weight.size(1), target_kernel_size,
                target_kernel_size)

            current_weight = target_weight

        return current_weight

    @staticmethod
    def _get_current_kernel_pos(source_kernel_size: int,
                                target_kernel_size: int) -> Tuple[int, int]:
        assert source_kernel_size > target_kernel_size, \
            '`source_kernel_size` must greater than `target_kernel_size`'

        center = source_kernel_size >> 1
        current_offset = target_kernel_size >> 1

        start_offset = center - current_offset
        end_offset = center + current_offset + 1

        return start_offset, end_offset

    def _get_padding(self, kernel_size) -> int:
        return kernel_size >> 1
