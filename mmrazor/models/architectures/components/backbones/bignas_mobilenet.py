# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Dict, List, Optional, Tuple, Union

import torch.utils.checkpoint as cp
from mmcls.models.backbones.base_backbone import BaseBackbone
from mmcls.models.builder import BACKBONES
from mmcls.models.utils import SELayer, make_divisible
from mmcv.cnn.bricks import (ConvModule, DropPath, build_activation_layer,
                             build_norm_layer)
from mmcv.runner import BaseModule, Sequential
from torch import nn
from torch.nn import Module
from torch.nn.modules.batchnorm import _BatchNorm

from mmrazor.models.mutables import DynamicKernelConv2d, DynamicSequential
from mmrazor.utils import master_only_print


class InvertResidual(BaseModule):
    """Mobilenet block for Searchable backbone.

    Args:
        kernel_size (int): Size of the convolving kernel.
        expand_ratio (int): The input channels' expand factor of the depthwise
             convolution.
        se_cfg (dict, optional): Config dict for se layer. Defaults to None,
            which means no se layer.
        conv_cfg (dict, optional): Config dict for convolution layer.
            Default: None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU').
        drop_path_rate (float): stochastic depth rate. Defaults to 0.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.

    Returns:
        Tensor: The output tensor.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int,
                 expand_ratio: int,
                 se_cfg: Optional[Dict] = None,
                 conv_cfg: Optional[Dict] = None,
                 dynamic_conv_cfg: Optional[Dict] = dict(
                     dynamic_kernel_size=(3, 5)),
                 norm_cfg: Dict = dict(type='BN'),
                 act_cfg: Dict = dict(type='ReLU'),
                 drop_path_rate: float = 0.,
                 with_cp: bool = False,
                 **kwargs: Any):

        super().__init__(**kwargs)
        self.with_res_shortcut = (stride == 1 and in_channels == out_channels)
        assert stride in [1, 2]
        self.stride = stride
        self.conv_cfg = conv_cfg
        self.dynamic_conv_cfg = dynamic_conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.with_cp = with_cp
        self.drop_path = DropPath(
            drop_path_rate) if drop_path_rate > 0 else nn.Identity()
        self.with_se = se_cfg is not None
        self.mid_channels = in_channels * expand_ratio
        self.with_expand_conv = (self.mid_channels != in_channels)

        if self.with_se:
            assert isinstance(se_cfg, dict)

        if self.with_expand_conv:
            self.expand_conv = ConvModule(
                in_channels=in_channels,
                out_channels=self.mid_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)

        if self.dynamic_conv_cfg:

            act_cfg_ = act_cfg.copy()
            if act_cfg_['type'] not in [
                    'Tanh', 'PReLU', 'Sigmoid', 'HSigmoid', 'Swish'
            ]:
                act_cfg_.setdefault('inplace', True)
            activate = build_activation_layer(act_cfg_)

            self.depthwise_conv = Sequential(
                DynamicKernelConv2d(
                    in_channels=self.mid_channels,
                    out_channels=self.mid_channels,
                    stride=stride,
                    dilation=1,
                    groups=self.mid_channels,
                    bias=False,
                    **dynamic_conv_cfg),
                build_norm_layer(norm_cfg, self.mid_channels)[1], activate)

        else:
            self.depthwise_conv = ConvModule(
                in_channels=self.mid_channels,
                out_channels=self.mid_channels,
                kernel_size=3,
                stride=self.stride,
                padding=1,
                groups=self.mid_channels,
                conv_cfg=self.conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)

        # TODO
        # should be searchable
        if self.with_se:
            self.se = SELayer(self.mid_channels, **se_cfg)

        self.linear_conv = ConvModule(
            in_channels=self.mid_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None)

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): The input tensor.
        Returns:
            torch.Tensor: The output tensor.
        """

        def _inner_forward(x):
            out = x

            if self.with_expand_conv:
                out = self.expand_conv(out)

            out = self.depthwise_conv(out)

            if self.with_se:
                out = self.se(out)

            out = self.linear_conv(out)

            if self.with_res_shortcut:
                return x + self.drop_path(out)
            else:
                return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        return out

    def init_weights(self) -> None:
        super().init_weights()

        if not self.with_res_shortcut:
            return

        last_bn_layer: _BatchNorm = self.linear_conv.norm
        if last_bn_layer.affine:
            master_only_print(f'init {type(last_bn_layer).__name__} to zero!')
            nn.init.constant_(last_bn_layer.weight, 0)


@BACKBONES.register_module()
class BigNASMobileNet(BaseBackbone):
    """Searchable MobileNet backbone.

    Args:
        first_channels (int): Channel width of first ConvModule. Default: 32.
        last_channels (int): Channel width of last ConvModule. Default: 1200.
        widen_factor (float): Width multiplier, multiply number of
            channels in each layer by this amount. Default: 1.0.
        out_indices (None or Sequence[int]): Output from which stages.
            Default: (7, ).
        frozen_stages (int): Stages to be frozen (all param fixed).
            Default: -1, which means not freezing any parameters.
        conv_cfg (dict, optional): Config dict for convolution layer.
            Default: None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU6').
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Default: False.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
        init_cfg (dict | list[dict]): initialization configuration dict to
            define initializer. OpenMMLab has implemented 6 initializers
            including ``Constant``, ``Xavier``, ``Normal``, ``Uniform``,
            ``Kaiming``, and ``Pretrained``.
    """

    # Parameters to build layers. 6 parameters are needed to construct a
    # layer, from left to right: expand_ratio, channel, max_blocks,
    # dynamic_blocks, dynamic_kernel_size, stride.
    arch_settings = [[1, 24, 2, list(range(1, 3)), (3, ), 1],
                     [6, 32, 3, list(range(2, 4)), (3, 5), 2],
                     [6, 48, 3, list(range(2, 4)), (3, 5), 2],
                     [6, 88, 4, list(range(2, 5)), (3, 5), 2],
                     [6, 128, 6, list(range(2, 7)), (3, 5), 1],
                     [6, 216, 6, list(range(2, 7)), (3, 5), 2],
                     [6, 352, 2, list(range(1, 3)), (3, 5), 1]]

    def __init__(
        self,
        first_channels: int = 40,
        last_channels: int = 1408,
        widen_factor: float = 1.,
        out_indices: Tuple[int] = (7, ),
        frozen_stages: int = -1,
        conv_cfg: Optional[Dict] = None,
        dynamic_conv_cfg: Optional[Dict] = dict(dynamic_kernel_size=(3, 5)),
        norm_cfg: Dict = dict(type='BN'),
        se_cfg: Optional[Dict] = None,
        act_cfg: Dict = dict(type='ReLU6'),
        norm_eval: bool = False,
        with_cp: bool = False,
        init_cfg: Union[Dict, List[Dict]] = [
            dict(type='Kaiming', layer=['Conv2d']),
            dict(type='Constant', val=1, layer=['_BatchNorm', 'GroupNorm'])
        ]):  # noqa: E125
        super().__init__(init_cfg)

        self.widen_factor = widen_factor
        self.out_indices = out_indices
        for index in out_indices:
            if index not in range(0, 8):
                raise ValueError('the item in out_indices must in '
                                 f'range(0, 8). But received {index}')

        if frozen_stages not in range(-1, 8):
            raise ValueError('frozen_stages must be in range(-1, 8). '
                             f'But received {frozen_stages}')
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.dynamic_conv_cfg = dynamic_conv_cfg
        self.norm_cfg = norm_cfg
        self.se_cfg = se_cfg
        self.act_cfg = act_cfg
        self.norm_eval = norm_eval
        self.with_cp = with_cp

        self.in_channels = make_divisible(first_channels * widen_factor, 8)

        self.conv1 = ConvModule(
            in_channels=3,
            out_channels=self.in_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.layers = []

        for i, layer_cfg in enumerate(self.arch_settings):
            expand_ratio, channel, num_blocks, \
                dynamic_blocks, dynamic_kernel_size, stride = layer_cfg
            out_channels = make_divisible(channel * widen_factor, 8)
            inverted_res_layer = self.make_layer(
                out_channels=out_channels,
                num_blocks=num_blocks,
                stride=stride,
                expand_ratio=expand_ratio,
                dynamic_blokcs=dynamic_blocks,
                dynamic_kernel_size=dynamic_kernel_size)

            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, inverted_res_layer)
            self.layers.append(layer_name)

        if widen_factor > 1.0:
            self.out_channel = int(last_channels * widen_factor)
        else:
            self.out_channel = last_channels

        layer = ConvModule(
            in_channels=self.in_channels,
            out_channels=self.out_channel,
            kernel_size=1,
            stride=1,
            padding=0,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.add_module('conv2', layer)
        self.layers.append('conv2')

    def make_layer(self, out_channels: int, num_blocks: int,
                   dynamic_blokcs: list, dynamic_kernel_size: list,
                   stride: int, expand_ratio: int) -> Module:
        """Stack InvertedResidual blocks to build a layer for MobileNetV2.

        Args:
            out_channels (int): out_channels of block.
            num_blocks (int): number of blocks.
            stride (int): stride of the first block. Default: 1
            expand_ratio (int): Expand the number of channels of the
                hidden layer in InvertedResidual by this ratio. Default: 6.
        """
        layers = []
        dynamic_conv_cfg = dict(dynamic_kernel_size=dynamic_kernel_size)
        for i in range(num_blocks):
            if i >= 1:
                stride = 1

            layers.append(
                InvertResidual(
                    in_channels=self.in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    expand_ratio=expand_ratio,
                    se_cfg=self.se_cfg,
                    conv_cfg=self.conv_cfg,
                    dynamic_conv_cfg=dynamic_conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                    with_cp=self.with_cp))

            self.in_channels = out_channels

        return DynamicSequential(*layers, dynamic_length=dynamic_blokcs)

    def forward(self, x):
        """Forward computation.

        Args:
            x (tensor | tuple[tensor]): x could be a Torch.tensor or a tuple of
                Torch.tensor, containing input data for forward computation.
        """
        x = self.conv1(x)

        outs = []
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)

        return tuple(outs)

    def _freeze_stages(self):
        """Freeze params not to update in the specified stages."""
        if self.frozen_stages >= 0:
            for param in self.conv1.parameters():
                param.requires_grad = False
        for i in range(1, self.frozen_stages + 1):
            layer = getattr(self, f'layer{i}')
            layer.eval()
            for param in layer.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        """Set module status before forward computation.

        Args:
            mode (bool): Whether it is train_mode or test_mode
        """
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()
