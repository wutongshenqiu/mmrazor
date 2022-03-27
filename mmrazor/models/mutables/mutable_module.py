# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from mmcv.runner import BaseModule


class MutableModule(BaseModule, metaclass=ABCMeta):
    """Base class for ``MUTABLES``. Searchable module for building searchable
    architecture in NAS. It mainly consists of module and mask, and achieving
    searchable function by handling mask.

    Args:
        space_id (str): Used to index ``Placeholder``, it is one and only index
            for each ``Placeholder``.
        num_chosen (str): The number of chosen ``OPS`` in the ``MUTABLES``.
        init_cfg (dict): Init config for ``BaseModule``.
    """

    def __init__(self,
                 space_id: str,
                 num_chosen: int = 1,
                 init_cfg: Optional[Dict] = None,
                 **kwargs) -> None:
        super(MutableModule, self).__init__(init_cfg)
        self.space_id = space_id
        self.num_chosen = num_chosen

    @abstractmethod
    def forward(self, x: Union[torch.Tensor, Tuple[torch.Tensor]]) -> Any:
        """Forward computation.

        Args:
            x (tensor | tuple[tensor]): x could be a Torch.tensor or a tuple of
                Torch.tensor, containing input data for forward computation.
        """
        pass

    @abstractmethod
    def build_choices(self, cfg: Dict) -> None:
        """Build all chosen ``OPS`` used to combine ``MUTABLES``, and the
        choices will be sampled.

        Args:
            cfg (dict): The config for the choices.
        """
        pass

    def build_choice_mask(self) -> torch.Tensor:
        """Generate the choice mask for the choices of ``MUTABLES``.

        Returns:
            torch.Tensor: Init choice mask. Its elements' type is bool.
        """
        if torch.cuda.is_available():
            return torch.ones(self.num_choices).bool().cuda()
        else:
            return torch.ones(self.num_choices).bool()

    def set_choice_mask(self, mask: torch.Tensor) -> None:
        """Use the mask to update the choice mask.

        Args:
            mask (torch.Tensor): Choice mask specified to update the choice
                mask.
        """
        # TODO
        # size(0) to shape?
        assert self.choice_mask.shape == mask.shape
        self.choice_mask = mask, 'Newer mask should have the same shape ' \
            f'as original, but got newer: {mask.shape}, ' \
            f'original: {self.choice_mask.shape}'

    @property
    def num_choices(self) -> int:
        """The number of the choices.

        Returns:
            int: the length of the choices.
        """
        return len(self.choices)

    @property
    def choice_names(self) -> Tuple[str]:
        """The choices' names.

        Returns:
            tuple: The keys of the choices.
        """
        assert isinstance(self.choices, nn.ModuleDict), \
            'candidates must be nn.ModuleDict.'
        return tuple(self.choices.keys())

    @property
    def choice_modules(self) -> Tuple[nn.Module]:
        """The choices' modules.

        Returns:
            tuple: The values of the choices.
        """
        assert isinstance(self.choices, nn.ModuleDict), \
            'candidates must be nn.ModuleDict.'
        return tuple(self.choices.values())

    def build_space_mask(self) -> torch.Tensor:
        """Generate the space mask for the search spaces of ``MUTATORS``.

        Returns:
            torch.Tensor: Init choice mask. Its elements' type is float.
        """
        if torch.cuda.is_available():
            return torch.ones(self.num_choices).cuda() * 1.0
        else:
            return torch.ones(self.num_choices) * 1.0

    # TODO
    # set maybe better than list when using `in` operation
    def export(self, chosen: List[str]) -> None:
        """Delete not chosen ``OPS`` in the choices.

        Args:
            chosen (list[str]): Names of chosen ``OPS``.
        """
        for name in self.choice_names:
            if name not in chosen:
                self.choices.pop(name)
