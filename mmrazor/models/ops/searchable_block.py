# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Dict, Optional

import torch.utils.checkpoint as cp
from mmcls.models.utils import SELayer
from mmcv.cnn.bricks import (ConvModule, DropPath, build_activation_layer,
                             build_norm_layer)
from mmcv.runner import BaseModule
from torch import nn

from mmrazor.models.architectures import Placeholder
from ..builder import OPS


@OPS.register_module()
class SearchableMBBlock(BaseModule):
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

    def __init__(
            self,
            # HACK
            space_id_prefix: str,
            group: str,
            in_channels: int,
            out_channels: int,
            stride: int,
            expand_ratio: int,
            se_cfg: Optional[Dict] = None,
            conv_cfg: Optional[Dict] = None,
            norm_cfg: Dict = dict(type='BN'),
            act_cfg: Dict = dict(type='ReLU'),
            drop_path_rate: float = 0.,
            with_cp: bool = False,
            **kwargs: Any):

        super().__init__(**kwargs)
        self.with_res_shortcut = (stride == 1 and in_channels == out_channels)
        assert stride in [1, 2]
        self.conv_cfg = conv_cfg
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

        act_cfg_ = act_cfg.copy()
        if act_cfg_['type'] not in [
                'Tanh', 'PReLU', 'Sigmoid', 'HSigmoid', 'Swish'
        ]:
            act_cfg_.setdefault('inplace', True)
        activate = build_activation_layer(act_cfg_)
        self.depthwise_conv = nn.Sequential(
            Placeholder(
                group=group,
                space_id=f'{space_id_prefix}_depthwise_conv2d',
                choice_args=dict(
                    in_channels=self.mid_channels,
                    out_channels=self.mid_channels,
                    stride=stride,
                    padding=0,
                    dilation=1,
                    groups=self.mid_channels,
                    bias=False)),
            build_norm_layer(norm_cfg, self.mid_channels)[1], activate)

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
