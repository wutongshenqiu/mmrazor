# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Dict, Tuple, Union

import torch.nn.functional as F
from mmcls.models.builder import HEADS
from mmcls.models.heads import ClsHead
from mmcv.cnn import build_activation_layer, build_norm_layer
from mmcv.runner import Sequential
from mmcv.utils import Registry
from torch import Tensor, nn

MMRAZOR_HEADS = Registry('mmrazor heads', parent=HEADS)


@MMRAZOR_HEADS.register_module()
class LinearClsHead(ClsHead):

    def __init__(self,
                 in_channels: int,
                 num_classes: int,
                 dropout_rate: float = 0.,
                 norm_cfg: Dict = None,
                 act_cfg: Dict = None,
                 init_cfg: Dict = None,
                 *args: Any,
                 **kwargs: Any):
        super().__init__(init_cfg=init_cfg, *args, **kwargs)

        self.in_channels = in_channels
        self.num_classes = num_classes

        if self.num_classes <= 0:
            raise ValueError(
                f'num_classes={num_classes} must be a positive integer')

        layers = []
        # same order as once-for-all
        # https://github.com/mit-han-lab/once-for-all/blob/4451593507b0f48a7854763adfe7785705abdd78/ofa/utils/layers.py#L293
        # dropout + linear + norm + act
        if dropout_rate > 0:
            layers.append(nn.Dropout(p=dropout_rate))
        layers.append(nn.Linear(in_channels, num_classes))
        if norm_cfg is not None:
            layers.append(build_norm_layer(norm_cfg, num_classes)[1])
        if act_cfg is not None:
            layers.append(build_activation_layer(act_cfg))

        self._layers = Sequential(*layers)

    def pre_logits(self, x: Union[Tensor, Tuple[Tensor]]) -> Tensor:
        if isinstance(x, tuple):
            x = x[-1]

        return x

    @property
    def fc(self) -> Sequential:
        return self._layers

    def forward_train(self, x: Union[Tensor, Tuple[Tensor]], gt_label: Tensor,
                      **kwargs: Any) -> Dict:
        x = self.pre_logits(x)
        cls_score = self.fc(x)
        losses = self.loss(cls_score, gt_label, **kwargs)

        return losses

    def init_weights(self) -> None:
        self._layers.init_weights()

    def simple_test(self, x, softmax=True, post_process=True):
        """Inference without augmentation.

        Args:
            x (tuple[Tensor]): The input features.
                Multi-stage inputs are acceptable but only the last stage will
                be used to classify. The shape of every item should be
                ``(num_samples, in_channels)``.
            softmax (bool): Whether to softmax the classification score.
            post_process (bool): Whether to do post processing the
                inference results. It will convert the output to a list.

        Returns:
            Tensor | list: The inference results.

                - If no post processing, the output is a tensor with shape
                  ``(num_samples, num_classes)``.
                - If post processing, the output is a multi-dimentional list of
                  float and the dimensions are ``(num_samples, num_classes)``.
        """
        x = self.pre_logits(x)
        cls_score = self.fc(x)

        if softmax:
            pred = (
                F.softmax(cls_score, dim=1) if cls_score is not None else None)
        else:
            pred = cls_score

        if post_process:
            return self.post_process(pred)
        else:
            return pred
