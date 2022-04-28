# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Iterable, List, Tuple

import torch
from torch import Tensor, nn
from torch.nn import Conv2d
from torch.nn import functional as F

from mmrazor.utils import master_only_print
from ..builder import MUTABLES
from .base import DynamicMutable


@MUTABLES.register_module()
class DynamicKernelConv2d(DynamicMutable[int], Conv2d):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size_list: Iterable[int],
                 stride: int = 1,
                 padding: int = 0,
                 padding_mode: str = 'zeros',
                 dilation: int = 1,
                 groups: int = 1,
                 bias: bool = True) -> None:
        kernel_size_list = sorted(list(set(kernel_size_list)), reverse=True)
        self._kernel_size_list = kernel_size_list
        self.set_choice(self.max_choice)

        Conv2d.__init__(
            self=self,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=self.max_choice,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode)

        transform_matrix_name_list = []
        for i in range(self.num_choices - 1):
            source_kernel_size = self.choices[i]
            target_kernel_size = self.choices[i + 1]
            transform_matrix_name = self._get_transform_matrix_name(
                src=source_kernel_size, tar=target_kernel_size)
            transform_matrix_name_list.append(transform_matrix_name)
            self.register_parameter(
                name=transform_matrix_name,
                param=nn.Parameter(torch.eye(target_kernel_size**2)))
        self._transform_matrix_name_list = transform_matrix_name_list

        self._is_deployed = False

    @property
    def is_deployed(self) -> bool:
        return self._is_deployed

    @property
    def choices(self) -> List[int]:
        return self._kernel_size_list

    @property
    def current_choice(self) -> int:
        return self._current_choice

    def set_choice(self, choice: int) -> None:
        assert choice in self.choices, \
            f'`choice` must be in: {self.choices}, but got: {choice}'
        self._current_choice = choice

    @torch.no_grad()
    def deploy_subnet(self, subnet_config: Dict) -> None:
        if self.is_deployed:
            # TODO
            # warning
            return

        choice = self.get_subnet_choice(subnet_config)
        weight = self._get_weight(choice)
        padding = self._get_padding(choice)
        self.weight = nn.Parameter(weight)
        self.padding = padding

        for transform_matrix_name in self._transform_matrix_name_list:
            delattr(self, transform_matrix_name)
            master_only_print(
                f'delete transform matrix: {transform_matrix_name}')

        self._is_deployed = True

    def forward_deploy(self, x: Tensor) -> Tensor:
        return Conv2d.forward(self, x)

    def forward_sample(self, x: Tensor, choice: int) -> Tensor:
        weight = self._get_weight(choice)
        padding = self._get_padding(choice)

        return F.conv2d(
            input=x,
            weight=weight,
            bias=self.bias,
            stride=self.stride,
            padding=padding,
            dilation=self.dilation,
            groups=self.groups)

    @staticmethod
    def _get_transform_matrix_name(src: int, tar: int) -> str:
        return f'transform_matrix_{src}to{tar}'

    def _get_weight(self, kernel_size: int) -> torch.Tensor:
        if kernel_size == self.max_choice:
            return self.weight

        current_weight = self.weight[:, :, :, :]
        for i in range(self.num_choices - 1):
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

    @staticmethod
    def _get_padding(kernel_size: int) -> int:
        return kernel_size >> 1
