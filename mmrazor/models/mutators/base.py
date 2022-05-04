# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC, abstractmethod
from typing import Any

from mmcv.runner import BaseModule

from mmrazor.models.architectures.base import BaseArchitecture


class BaseMutator(ABC, BaseModule):
    """Base class for mutators."""

    @abstractmethod
    def prepare_from_supernet(self, supernet: BaseArchitecture) -> None:
        """Do some necessary preparations with supernet.

        Args:
            supernet (:obj:`torch.nn.Module`): The architecture to be used
                in your algorithm.
        """

    @property
    @abstractmethod
    def search_space(self) -> Any:
        ...
