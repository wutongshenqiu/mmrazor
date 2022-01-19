# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Dict

import torch

from mmrazor.models.builder import ALGORITHMS, build_pruner
from mmrazor.models.pruners import ResRepPruner
from .base import BaseAlgorithm


@ALGORITHMS.register_module()
class ResRep(BaseAlgorithm):
    # TODO: update docs

    def __init__(self, before_update_mask_iter: int, mask_interval: int,
                 **kwargs) -> None:
        super(ResRep, self).__init__(**kwargs)

        self._before_update_mask_iter = before_update_mask_iter
        self._mask_interval = mask_interval
        # TODO: use hook
        self.register_buffer('_current_iter', torch.LongTensor([0]))

    def _init_pruner(self, pruner: Dict) -> None:
        """Build registered pruners and make preparations.

        Args:
            pruner (dict): The registered pruner to be used
                in the algorithm.
        """
        if pruner is None:
            self.pruner = None
            return

        # judge whether our StructurePruner can prune the architecture
        try:
            pseudo_pruner = build_pruner(pruner)
            pseudo_architecture = copy.deepcopy(self.architecture)
            pseudo_pruner.prepare_from_supernet(pseudo_architecture)
            subnet_dict = pseudo_pruner.sample_subnet(pseudo_architecture)
            pseudo_pruner.set_subnet(subnet_dict)
            subnet_dict = pseudo_pruner.export_subnet()

            pseudo_pruner.deploy_subnet(pseudo_architecture, subnet_dict)
            pseudo_img = torch.randn(1, 3, 224, 224)
            pseudo_architecture.forward_dummy(pseudo_img)
        except RuntimeError:
            raise NotImplementedError('Our current StructurePruner does not '
                                      'support pruning this architecture. '
                                      'StructurePruner is not perfect enough '
                                      'to handle all the corner cases. We will'
                                      ' appreciate it if you create a issue.')

        self.pruner: ResRepPruner = build_pruner(pruner)
        self.pruner.prepare_from_supernet(self.architecture)

    def train_step(self, data: Dict, optimizer: torch.optim.Optimizer) -> Dict:
        """Train step function.

        This function implements the standard training iteration for resrep.

        Args:
            data (dict): Input data from dataloader.
            optimizer (:obj:`torch.optim.Optimizer`): The optimizer to
                accumulate gradient
        """
        self._update_iter(count=1)
        if self.iter > self._before_update_mask_iter \
                and self.iter % self._mask_interval == 0:
            self.pruner.module.update_mask(self.architecture.module)

        optimizer.zero_grad()
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)
        loss.backward()

        self.pruner.module.gradient_reset()

        optimizer.step()

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img'].data))

        return outputs

    # TODO: get lr in models
    # https://github.com/open-mmlab/mmcv/issues/1242
    @property
    def iter(self) -> int:
        return self._current_iter.item()

    def _update_iter(self, count: int = 1) -> None:
        self._current_iter += count
