# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Dict

import torch

from mmrazor.models.builder import ALGORITHMS
from mmrazor.models.distillers import SelfDistiller
from mmrazor.models.mutators import BigNASMutator
from mmrazor.models.pruners import RatioPruner
from mmrazor.models.utils import add_prefix
from .autoslim import AutoSlim


@ALGORITHMS.register_module()
class BigNAS(AutoSlim):
    pruner: RatioPruner
    mutator: BigNASMutator
    distiller: SelfDistiller
    num_sample_training: int

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        # assert self.pruner is not None, \
        #     'Pruner must be configured for BigNAS!'
        assert self.distiller is not None, \
            'Distiller must be configured for BigNAS!'
        assert self.mutator is not None, \
            'Mutator must be configured for BigNAS!'

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

        losses = dict()
        self._set_max_subnet()

        max_subnet_losses = self.distiller.exec_teacher_forward(
            self.architecture, data)
        losses.update(add_prefix(max_subnet_losses, 'max_subnet'))
        max_subnet_loss, _ = self._parse_losses(max_subnet_losses)
        max_subnet_loss.backward()

        self._set_min_subnet()
        self.distiller.exec_student_forward(self.architecture, data)
        min_subnet_losses = self.distiller.compute_distill_loss(data)
        losses.update(add_prefix(min_subnet_losses, 'min_subnet'))
        min_subnet_loss, _ = self._parse_losses(min_subnet_losses)
        min_subnet_loss.backward()

        for i in range(self.num_sample_training - 2):
            self._set_random_subnet()
            self.distiller.exec_student_forward(self.architecture, data)
            subnet_losses = self.distiller.compute_distill_loss(data)
            losses.update(add_prefix(subnet_losses, f'random_subnet{i + 1}'))

            subnet_loss, _ = self._parse_losses(subnet_losses)
            subnet_loss.backward()

        # TODO: clip grad norm
        optimizer.step()

        loss, log_vars = self._parse_losses(losses)
        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img'].data))

        return outputs

    def _set_min_subnet(self) -> None:
        """set minimum subnet in current search space."""
        self.pruner.set_min_channel()
        self.mutator.set_min_subnet()

    def _set_max_subnet(self) -> None:
        """set maximum subnet in current search space."""
        self.pruner.set_max_channel()
        self.mutator.set_max_subnet()

    def _set_random_subnet(self) -> None:
        """set random subnet in current search space."""
        subnet_dict = self.pruner.sample_subnet()
        self.pruner.set_subnet(subnet_dict)
        self.mutator.set_random_subnet()
