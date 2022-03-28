# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Tuple

import torch

from ..builder import MUTABLES
from .mixins import OrderedChoiceMixin
from .mutable_module import MutableModule


@MUTABLES.register_module()
class MutableSequential(MutableModule, OrderedChoiceMixin):

    def __init__(self, length_list: List[int], choices: torch.nn.Sequential,
                 **kwargs) -> None:
        super().__init__(**kwargs)

        assert len(length_list) >= 2, 'Sequential length should be greater ' \
            f'than 2, but got: {len(length_list)}'

        length_list = sorted(length_list)
        assert length_list[0] > 0
        assert length_list[-1] == len(choices), 'Max searchable length ' \
            f'should less than sequential length, ' \
            f'got max searchable length: {length_list[-1]}, ' \
            f'sequential length: {len(choices)}'

        self.choices = choices
        self._length_list = length_list
        # TODO
        # ugly
        self.choice_mask = self.build_choice_mask()
        self.choice_mask = self.max_choice_mask

        assert len(self.choice_mask.shape) == 1
        print(f'space id: {self.space_id} choice mask: {self.choice_mask}')

    def build_choices(self, cfg: Dict) -> None:
        pass

    @property
    def choice_names(self) -> Tuple[str]:
        return tuple(map(str, range(len(self._length_list))))

    @property
    def num_choices(self) -> int:
        return len(self._length_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        choice_idx = self.choice_mask.nonzero()[0].item()
        length = self._length_list[choice_idx]
        print(f'space id: {self.space_id}, current length: {length}, '
              f'max length: {self._length_list[-1]}')

        return self.choices[:self._length_list[choice_idx]](x)
