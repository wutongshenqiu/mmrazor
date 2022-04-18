# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict

import torch
import torch.distributed as dist

from mmrazor.models.builder import MUTATORS
from ..mutables import MutableModule, OrderedChoiceMixin
from .base import BaseMutator


@MUTATORS.register_module()
class BigNASMutator(BaseMutator):
    """A mutator for the one-shot NAS, which mainly provide some core functions
    of changing the structure of ``ARCHITECTURES``."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def prepare_from_supernet(self, supernet: torch.nn.Module) -> None:
        super().prepare_from_supernet(supernet)

        self._check_search_space()

    def _check_search_space(self) -> None:
        for _, space_info in self.search_spaces.items():
            modules = space_info['modules']
            if len(modules) != 1:
                raise RuntimeError('Expect number of modules to be 1, '
                                   f'but got: {len(modules)}')
            if not isinstance(modules[0], OrderedChoiceMixin):
                raise TypeError('Expect module to be instance of '
                                f'{OrderedChoiceMixin.__name__}, '
                                f'but got: {type(modules[0].__name__)}')

    def _set_subnet(self, subnet_dict: Dict[str, torch.Tensor]) -> None:
        """Setting subnet in the supernet based on the result of
        ``sample_subnet`` by changing the flag: ``in_subnet``, which is easy to
        implement some operations for subnet, such as ``forward``, calculate
        flops and so on.

        Args:
            subnet_dict (dict): Record the information to build the subnet
                from the supernet,
                its keys are the properties ``space_id`` of placeholders in the
                mutator's search spaces,
                its values are masks.
        """
        for space_id, space_info in self.search_spaces.items():
            choice_mask = subnet_dict[space_id]
            for module in space_info['modules']:
                module: MutableModule
                module.set_choice_mask(choice_mask)

    def _sample_subnet(self, sample_policy: str) -> Dict[str, torch.Tensor]:
        """Sample random subnet in current search space.

        Returns:
            Dict[str, torch.Tensor]: _description_
        """
        valid_sample_policies = {'min', 'max', 'random'}
        if sample_policy not in valid_sample_policies:
            raise ValueError(f'Invalid sample_policy: {sample_policy}, '
                             f'expect one of {valid_sample_policies}')

        def get_choice_mask(module: OrderedChoiceMixin) -> torch.Tensor:
            if sample_policy == 'min':
                return module.min_choice_mask
            elif sample_policy == 'max':
                return module.max_choice_mask
            else:
                return module.random_choice_mask

        subnet_dict = dict()
        for space_id, space_info in self.search_spaces.items():
            choice_mask = get_choice_mask(space_info['modules'][0])
            # TODO
            # broadcast choice_mask?
            if dist.is_available() and dist.is_initialized():
                dist.broadcast(choice_mask, src=0)

            subnet_dict[space_id] = choice_mask

        return subnet_dict

    def _sample_set_subnet(self, sample_policy: str) -> None:
        subnet_dict = self._sample_subnet(sample_policy)
        self._set_subnet(subnet_dict)

    def set_max_subnet(self) -> None:
        """set maximum subnet in current search space."""
        self._sample_set_subnet('max')

    def set_min_subnet(self) -> None:
        """set minimum subnet in current search space."""
        self._sample_set_subnet('min')

    def set_random_subnet(self) -> None:
        """set random subnet in current search space."""
        self._sample_set_subnet('random')

    def export_subnet(self) -> Dict:
        """Export mutator subnet config.

        Returns:
            Dict: _description_
        """
        subnet_dict = dict()
        for space_id, space_info in self.search_spaces.items():
            for module in space_info['modules']:
                module: MutableModule
                subnet_dict[space_id] = module.current_choice_map

        return subnet_dict

    def deploy_subnet(self, supernet: torch.nn.Module,
                      subnet_dict: Dict) -> None:
        for space_id, space_info in self.search_spaces.items():
            for module in space_info['modules']:
                module: MutableModule
                choice_map = subnet_dict[space_id]
                module.set_choice_map(choice_map)
