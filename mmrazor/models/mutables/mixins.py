# Copyright (c) OpenMMLab. All rights reserved.
import random
from abc import ABC, abstractmethod
from typing import List

import torch


class OrderedChoiceMixin:
    """A mixin that provided useful methods when choices are ordered."""
    choice_mask: torch.Tensor
    num_choices: int

    @property
    def min_choice_mask(self) -> torch.Tensor:
        """Choice mask with minimum squential length."""
        choice_mask = torch.zeros_like(self.choice_mask)
        choice_mask[0] = 1

        return choice_mask

    @property
    def max_choice_mask(self) -> torch.Tensor:
        """Choice mask with maximum squential length."""
        choice_mask = torch.zeros_like(self.choice_mask)
        choice_mask[-1] = 1

        return choice_mask

    @property
    def random_choice_mask(self) -> torch.Tensor:
        """Choice mask with random sequential length in length list."""
        choice_mask = torch.zeros_like(self.choice_mask)
        choice_idx = random.randint(0, self.num_choices - 1)
        choice_mask[choice_idx] = 1

        return choice_mask


class MutableMixin(ABC):

    @abstractmethod
    def deploy(self, chosen) -> None:
        pass

    @property
    @abstractmethod
    def deployed(self) -> bool:
        pass

    @property
    @deployed.setter
    @abstractmethod
    def deployed(self, value) -> None:
        pass

    @property
    @abstractmethod
    def num_choices(self) -> int:
        """The number of the choices.

        Returns:
            int: the length of the choices.
        """
        pass

    @property
    @abstractmethod
    def choices(self) -> List:
        pass


class OneShotMixin(MutableMixin):

    @property
    @abstractmethod
    def choice_probs(self) -> List:
        pass

    @property
    @choice_probs.setter
    @abstractmethod
    def choice_probs(self, value: List) -> None:
        pass

    def forward(self, x, sampled=None):
        if self.deployed:
            assert sampled is None
            return self.forward_deploy(x)
        elif sampled:
            return self.forward_sample(x, sampled)

    @abstractmethod
    def forward_deploy(self, x):
        pass

    @abstractmethod
    def forward_sample(self, x, sampled):
        pass


class DiffentiableMixin(MutableMixin):

    def forward(self, x, arch_params=None):
        if self.deployed:
            assert arch_params is None
            return self.forward_deploy(x)
        elif arch_params:
            return self.forward_arch_params(x, arch_params)

    @abstractmethod
    def forward_deploy(self, x):
        pass

    @abstractmethod
    def forward_arch_params(self, x, arch_params: torch.nn.Parameter):
        pass


class DynamicMixin(OneShotMixin):

    @property
    @abstractmethod
    def min_choice(self):
        pass

    @property
    @abstractmethod
    def max_choice(self):
        pass

    def forward(self, x, sampled=None):
        if self.deployed:
            return self.forward_deploy(x)

        elif sampled in self.choices:
            return self.forward_sample(x, sampled)

        elif sampled is None:
            return self.forward_sample(x, self.max_choice)
