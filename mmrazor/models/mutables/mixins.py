# Copyright (c) OpenMMLab. All rights reserved.
import random

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
