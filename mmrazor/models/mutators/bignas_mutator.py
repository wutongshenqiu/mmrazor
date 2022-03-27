# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict

import torch
import torch.distributed as dist

from mmrazor.models.builder import MUTATORS
from ..mutables import MutableModule
from .base import BaseMutator


@MUTATORS.register_module()
class BigNASMutator(BaseMutator):
    """A mutator for the one-shot NAS, which mainly provide some core functions
    of changing the structure of ``ARCHITECTURES``."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def get_random_mask(space_info: Dict) -> torch.Tensor:
        """Generate random mask for randomly sampling.

        Args:
            space_info (dict): Record the information of the space need
                to sample.
            searching (bool): Whether is in search stage.

        Returns:
            torch.Tensor: Random mask generated.
        """
        space_mask = space_info['space_mask']
        num_chosen = space_info['num_chosen']
        assert num_chosen <= space_mask.size()[0]
        choice_idx = torch.multinomial(space_mask, num_chosen)
        choice_mask = torch.zeros_like(space_mask)
        choice_mask[choice_idx] = 1
        if dist.is_available() and dist.is_initialized():
            dist.broadcast(choice_mask, src=0)
        return choice_mask

    def sample_subnet(self) -> Dict[str, torch.Tensor]:
        """Random sample subnet by random mask.

        Returns:
            dict: Record the information to build the subnet from the supernet,
                its keys are the properties ``space_id`` of placeholders in the
                mutator's search spaces,
                its values are random mask generated.
        """
        subnet_dict = dict()
        for space_id, space_info in self.search_spaces.items():
            subnet_dict[space_id] = self.get_random_mask(space_info)
        return subnet_dict

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

    def set_max_subnet(self) -> None:
        """set maximum subnet in current search space."""

        raise NotImplementedError

    def set_min_subnet(self) -> None:
        """set minimum subnet in current search space."""

        raise NotImplementedError

    def set_random_subnet(self) -> None:
        """set random subnet in current search space."""

        raise NotImplementedError
