# Copyright (c) OpenMMLab. All rights reserved.
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import mmcv
import torch
from torch.nn import Dropout, functional

from mmrazor.models.architectures.base import BaseArchitecture
from mmrazor.models.builder import ALGORITHMS
from mmrazor.models.distillers import SelfDistiller
from mmrazor.models.mutators import DynamicMutator
from mmrazor.models.pruners import RangePruner
from mmrazor.models.utils import add_prefix
from mmrazor.utils import master_only_print
from .base import BaseAlgorithm


class _InputResizer:
    valid_interpolation_type = {
        'nearest', 'linear', 'bilinear', 'bicubic', 'trilinear', 'area',
        'nearest-exact'
    }

    def __init__(self,
                 shape_list: List[Union[Sequence[int], int]],
                 interpolation_type: str = 'bicubic',
                 align_corners: bool = False,
                 scale_factor: Optional[Union[float, List[float]]] = None,
                 recompute_scale_factor: Optional[bool] = None) -> None:
        self.set_shape_list(shape_list)
        self.set_max_shape()
        self.set_interpolation_type(interpolation_type)

        self._scale_factor = scale_factor
        self._align_corners = align_corners
        self._recompute_scale_factor = recompute_scale_factor

    @property
    def current_interpolation_type(self) -> str:
        return self._current_interpolation_type

    def set_interpolation_type(self, interpolation_type: str) -> None:
        if interpolation_type not in self.valid_interpolation_type:
            raise ValueError(
                'Expect `interpolation_type` be '
                f'one of {self.valid_interpolation_type}, but got: '
                f'{interpolation_type}')

        self._current_interpolation_type = interpolation_type

    def set_shape_list(self, shape_list: List[Union[Sequence[int],
                                                    int]]) -> None:
        tuple_shape_list = []
        for shape in shape_list:
            if isinstance(shape, int):
                shape = (shape, shape)
            if len(shape) != 2:
                raise ValueError('Length of shape must be 2, '
                                 f'but got: {len(shape)}')
            tuple_shape_list.append(tuple(shape))

        self._shape_list: List[Tuple[int]] \
            = sorted(tuple_shape_list, key=lambda x: x[0] * x[1])
        self._shape_set = set(self._shape_list)

    def set_shape(self, shape: Union[int, Tuple[int]]) -> None:
        if isinstance(shape, int):
            shape = (shape, shape)
        if shape not in self._shape_set:
            raise ValueError(f'Expect shape to be one of: {self._shape_list} '
                             f'but got: {shape}')
        self._current_shape = shape

    @property
    def current_shape(self) -> Tuple[int]:
        return self._current_shape

    @property
    def shape_list(self) -> List[Tuple[int]]:
        return self._shape_list.copy()

    def set_max_shape(self) -> None:
        self.set_shape(self._shape_list[-1])

    def set_min_shape(self) -> None:
        self.set_shape(self._shape_list[0])

    def set_random_shape(self) -> None:
        random_shape = random.choice(self._shape_list)
        self.set_shape(random_shape)

    def resize(self, x: torch.Tensor) -> torch.Tensor:
        return functional.interpolate(
            input=x,
            size=self.current_shape,
            mode=self.current_interpolation_type,
            scale_factor=self._scale_factor,
            align_corners=self._align_corners,
            recompute_scale_factor=self._recompute_scale_factor)


@ALGORITHMS.register_module()
class BigNAS(BaseAlgorithm):
    pruner: RangePruner
    mutator: DynamicMutator
    distiller: SelfDistiller
    num_sample_training: int
    architecture: BaseArchitecture

    def __init__(self,
                 resizer_config: Optional[Dict] = None,
                 is_supernet_training: bool = False,
                 channel_cfg_path: Optional[str] = None,
                 **kwargs: Any) -> None:
        super().__init__(**kwargs)

        self._is_supernet_training = is_supernet_training
        if is_supernet_training:
            if resizer_config is None:
                raise ValueError('`resizer_config` must be configured when '
                                 'training supernet')
            self._resizer = _InputResizer(**resizer_config)
        else:
            if channel_cfg_path is not None:
                channel_cfg = mmcv.fileio.load(channel_cfg_path)
                self.pruner.deploy_subnet(self.architecture, channel_cfg)

        # assert self.pruner is not None, \
        #     'Pruner must be configured for BigNAS!'
        # assert self.distiller is not None, \
        #     'Distiller must be configured for BigNAS!'
        # assert self.mutator is not None, \
        #     'Mutator must be configured for BigNAS!'

    def train_step(self, data: Dict, optimizer: torch.optim.Optimizer) -> Dict:
        """Train step function.

        This function implements the standard training iteration for
        autoslim pretraining and retraining.

        Args:
            data (dict): Input data from dataloader.
            optimizer (:obj:`torch.optim.Optimizer`): The optimizer to
                accumulate gradient
        """
        optimizer.zero_grad()

        if self._is_supernet_training:
            losses = self._train_supernet_step(data)
        else:
            losses = self._retrain_step(data)

        # TODO: clip grad norm
        optimizer.step()

        loss, log_vars = self._parse_losses(losses)
        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img'].data))

        return outputs

    def _retrain_step(self, data: Dict) -> Dict:
        model_losses = self(**data)
        losses = add_prefix(model_losses, 'retrain_subnet')
        model_loss, _ = self._parse_losses(model_losses)
        model_loss.backward()

        return losses

    def _train_supernet_step(self, data: Dict) -> Dict:
        losses = dict()
        original_img = data['img'].clone()

        self._set_max_subnet()
        self._train_dropout(True)
        max_img = self._resizer.resize(original_img)
        data['img'] = max_img

        max_subnet_losses = self.distiller.exec_teacher_forward(
            self.architecture, data)
        losses.update(add_prefix(max_subnet_losses, 'max_subnet'))
        max_subnet_loss, _ = self._parse_losses(max_subnet_losses)
        max_subnet_loss.backward()

        self._set_min_subnet()
        self._train_dropout(False)

        min_img = self._resizer.resize(original_img)
        data['img'] = min_img

        self.distiller.exec_student_forward(self.architecture, data)
        min_subnet_losses = self.distiller.compute_distill_loss(data)
        losses.update(add_prefix(min_subnet_losses, 'min_subnet'))
        min_subnet_loss, _ = self._parse_losses(min_subnet_losses)
        min_subnet_loss.backward()

        for i in range(self.num_sample_training - 2):
            self._set_random_subnet()

            random_img = self._resizer.resize(original_img)
            data['img'] = random_img

            self.distiller.exec_student_forward(self.architecture, data)
            subnet_losses = self.distiller.compute_distill_loss(data)
            losses.update(add_prefix(subnet_losses, f'random_subnet{i + 1}'))

            subnet_loss, _ = self._parse_losses(subnet_losses)
            subnet_loss.backward()

    def _set_min_subnet(self) -> None:
        """set minimum subnet in current search space."""
        self.pruner.set_min_channel()
        self.mutator.set_subnet(self.mutator.min_subnet)
        self._resizer.set_min_shape()

    def _set_max_subnet(self) -> None:
        """set maximum subnet in current search space."""
        self.pruner.set_max_channel()
        self.mutator.set_subnet(self.mutator.max_subnet)
        self._resizer.set_max_shape()

    def _set_random_subnet(self) -> None:
        """set random subnet in current search space."""
        self.pruner.set_random_channel()
        self.mutator.set_subnet(self.mutator.random_subnet)
        self._resizer.set_random_shape()

    def _train_dropout(self, mode: bool = True) -> None:
        for name, module in self.architecture.named_modules():
            if isinstance(module, Dropout):
                master_only_print(f'set mode of `{name}` to: {mode}')
                module.train(mode=mode)
