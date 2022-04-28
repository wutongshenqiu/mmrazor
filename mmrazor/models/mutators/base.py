# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC, abstractmethod
from typing import Dict, Optional

from mmcv.runner import BaseModule
from torch.nn import Module


class BaseMutator(ABC, BaseModule):
    """Base class for mutators."""

    def __init__(self, init_cfg: Optional[Dict] = None) -> None:
        super().__init__(init_cfg=init_cfg)

    def prepare_from_supernet(self, supernet: Module) -> None:
        """Implement some preparatory work based on supernet, including
        ``convert_placeholder`` and ``build_search_spaces``.

        Args:
            supernet (:obj:`torch.nn.Module`): The architecture to be used
                in your algorithm.
        """

        self.search_space = self.build_search_space(supernet)

    @abstractmethod
    def build_search_space(self, supernet: Module) -> Dict:
        """Build a search space from the supernet.

        Args:
            supernet (:obj:`torch.nn.Module`): The architecture to be used
                in your algorithm.

        Returns:
            dict: To collect some information about ``MutableModule`` in the
                supernet.
        """
        pass
