# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple

import torch
from mmcv.cnn import ConvModule
from torch import Tensor
from torch.nn import Conv2d
from torch.nn import functional as F
from torch.nn.parameter import Parameter

from mmrazor.utils import master_only_print
from ..builder import OPS
from .base import BaseDynamicOP

_DYNAMIC_CONV_OPS = {'DynamicConv2d'}
_DYNAMIC_PLUGIN_LAYERS = {'DynamicConvModule'}

_DYNAMIC_OPS = _DYNAMIC_CONV_OPS | _DYNAMIC_PLUGIN_LAYERS


def _register_dynamic_ops() -> None:
    from mmcv.cnn import CONV_LAYERS, PLUGIN_LAYERS

    for op in _DYNAMIC_CONV_OPS:
        CONV_LAYERS.register_module(op, module=OPS.get(op))

    for op in _DYNAMIC_PLUGIN_LAYERS:
        PLUGIN_LAYERS.register_module(op, module=OPS.get(op))


@OPS.register_module()
class DynamicConv2d(BaseDynamicOP[int], Conv2d):
    choice_map_key: str = 'kernel_size'

    def __init__(self, choices: List[int], in_channels: int, out_channels: int,
                 **other_conv_kwargs) -> None:
        for c in choices:
            if c & 1 == 0:
                raise ValueError(f'Kernel size must be odd, bug got: {c}!')
        assert other_conv_kwargs.get('kernel_size') is None, \
            '`kernel_size` should contain in choices, ' \
            'but not a keyword parameter!'

        # TODO
        # dangerous when occurs `diamond inheritance`
        BaseDynamicOP.__init__(self, choices)

        self._current_choice = choices[-1]
        self._max_kernel_size = choices[-1]

        Conv2d.__init__(
            self,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=self._max_kernel_size,
            **other_conv_kwargs)

        assert self.padding_mode == 'zeros', 'Only support `zeros` ' \
            f'padding mode, but got: {self.padding_mode}'

        for i in range(len(choices) - 1, 0, -1):
            source_kernel_size = choices[i]
            target_kernel_size = choices[i - 1]
            transform_matrix_name = self._get_transform_matrix_name(
                src=source_kernel_size, tar=target_kernel_size)
            self.register_parameter(
                name=transform_matrix_name,
                param=Parameter(torch.eye(target_kernel_size**2)))

    def set_choice(self, choice: int) -> None:
        if choice not in self._choices:
            raise ValueError(f'New choice: {choice} not '
                             f'in previous choice list: {self._choices}')

        self._current_choice = choice

    @property
    def current_choice(self) -> int:
        return self._current_choice

    @staticmethod
    def _get_transform_matrix_name(src: int, tar: int) -> str:
        return f'transform_matrix_{src}to{tar}'

    def forward(self, input: Tensor) -> Tensor:
        weight = self._get_current_weight()
        padding = self._get_current_padding()

        return F.conv2d(
            input=input,
            weight=weight,
            bias=self.bias,
            stride=self.stride,
            padding=padding,
            dilation=self.dilation,
            groups=self.groups)

    def _get_current_weight(self) -> torch.Tensor:
        if self.current_choice == self._max_kernel_size:
            return self.weight

        current_weight = self.weight[:, :, :, :]
        for i in range(len(self._choices) - 1, 0, -1):
            source_kernel_size = self._choices[i]
            if source_kernel_size <= self.current_choice:
                break
            target_kernel_size = self._choices[i - 1]
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
            weight = current_weight[:, :, start_offset:end_offset,
                                    start_offset:end_offset]
            weight = weight.reshape(-1, target_kernel_size**2)
            weight = F.linear(weight, transform_matrix)
            weight = weight.reshape(
                self.weight.size(0), self.weight.size(1), target_kernel_size,
                target_kernel_size)

            current_weight = weight

        return weight

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

    def _get_current_padding(self) -> int:
        return self.current_choice >> 1


@OPS.register_module()
class DynamicConvModule(BaseDynamicOP[int], ConvModule):
    conv: BaseDynamicOP[int]
    choice_map_key: str = 'kernel_size'

    def __init__(self, choices: List[int], **conv_module_kwargs) -> None:
        conv_cfg = conv_module_kwargs.get('conv_cfg')
        if conv_cfg is None or conv_cfg.get('type') not in _DYNAMIC_CONV_OPS:
            raise ValueError(
                f'`conv_cfg` must contain dynamic conv ops, '
                f'but got: {conv_cfg}, available ops: {_DYNAMIC_CONV_OPS}')
        conv_cfg['choices'] = choices
        master_only_print(f'choices: {choices}')
        master_only_print(f'conv module kwargs: {conv_module_kwargs}')
        BaseDynamicOP.__init__(self, choices)
        ConvModule.__init__(self, **conv_module_kwargs)

    def set_choice(self, choice: int) -> None:
        self.conv.set_choice(choice)

    @property
    def current_choice(self) -> int:
        return self.conv.current_choice


_register_dynamic_ops()
