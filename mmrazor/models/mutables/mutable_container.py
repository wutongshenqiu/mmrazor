# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Dict, Iterable, List, Optional

from mmcv.runner import Sequential
from torch import Tensor

from ..builder import MUTABLES
from .base import DynamicMutable


# TODO
# how to build?
@MUTABLES.register_module()
class DynamicSequential(DynamicMutable[int], Sequential):

    def __init__(self,
                 *args: Any,
                 length_list: Iterable[int],
                 init_cfg: Optional[Dict] = None) -> None:
        Sequential.__init__(self, *args, init_cfg=init_cfg)

        self._length_list = sorted(list(set(length_list)), reverse=True)
        assert self.max_choice == len(self), \
            'Max length should be the length of total sequential!'
        self.set_choice(self.max_choice)

        self._is_deployed = False

    @property
    def current_choice(self) -> int:
        return self._current_choice

    def set_choice(self, choice: int) -> None:
        assert choice in self.choices, \
            f'`choice` must be in: {self.choices}, but got: {choice}'
        self._current_choice = choice

    @property
    def choices(self) -> List[int]:
        return self._length_list

    @property
    def is_deployed(self) -> bool:
        return self._is_deployed

    def deploy_subnet(self, subnet_config: Dict) -> None:
        if self.is_deployed:
            return

        choice = self.get_subnet_choice(subnet_config)
        del self[choice:]

        self._is_deployed = True

    def forward_deploy(self, x: Tensor) -> Tensor:
        return Sequential.forward(self, x)

    def forward_sample(self, x: Tensor, choice: int) -> Tensor:
        for idx, module in enumerate(self):
            if idx < choice:
                x = module(x)
            else:
                break

        return x
