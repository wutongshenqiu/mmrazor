# Copyright (c) OpenMMLab. All rights reserved.
from torch import Tensor, nn

from .mixins import DynamicMixin


class DynamicSequential(DynamicMixin, nn.Sequential):

    def __init__(self, *args, dynamic_length) -> None:
        super().__init__(*args)
        assert isinstance(dynamic_length, (tuple, list))
        self.dynamic_length = sorted(list(dynamic_length), reverse=True)
        self.max_length = max(self.dynamic_length)

        self._deployed = False
        self._choice_probs = [1 / len(self.dynamic_length)] * len(
            self.dynamic_length)

    @property
    def choices(self):
        return self.dynamic_length

    @property
    def num_choices(self) -> int:
        return len(self.dynamic_length)

    @property
    def max_choice(self):
        max_choice = len(self._modules)
        assert max_choice == len(self._modules)
        return max_choice

    @property
    def min_choice(self):
        return min(self.dynamic_length)

    @property
    def choice_probs(self):
        assert sum(self._choice_probs) == 1
        return self._choice_probs

    @choice_probs.setter
    def choice_probs(self, value):
        self._choice_probs = value

    @property
    def deployed(self):
        return self._deployed

    @deployed.setter
    def deployed(self, value):
        self._deployed = value

    def deploy(self, chosen):
        for _ in range(self.max_length - chosen):
            del self[-1]
        self.deployed = True

    def forward_deploy(self, x):
        return nn.Sequential.forward(self, x)

    def forward_sample(self, input: Tensor, sampled=None) -> Tensor:
        for idx, module in enumerate(self):
            if idx < sampled:
                input = module(input)
            else:
                break
        return input
