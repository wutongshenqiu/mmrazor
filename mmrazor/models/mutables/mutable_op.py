# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmrazor.models.ops.base import CHOICE_TYPE, BaseDynamicOP
from ..builder import MUTABLES, build_op
from .mixins import OrderedChoiceMixin
from .mutable_module import MutableModule


@MUTABLES.register_module()
class DynamicOP(MutableModule, OrderedChoiceMixin):

    def __init__(self, choices: List[CHOICE_TYPE], dynamic_cfg: Dict,
                 choice_args: Dict, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        # `dynamic_cfg` has higher priority
        choice_args.update(dynamic_cfg, choices=choices)
        op = build_op(choice_args)
        assert isinstance(op, BaseDynamicOP), \
            f'OP must be dynamic, but got type: {type(op).__name__}'

        # the choices is already in order
        self.choices = op.choices()
        self._dynamic_op = op

        self.choice_mask = self.build_choice_mask()
        self.choice_mask = self.max_choice_mask

    def build_choices(self, cfg: Dict) -> None:
        pass

    @property
    def choice_names(self) -> Tuple[str]:
        return tuple(map(str, range(len(self.choices))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        choice_idx = self.choice_mask.nonzero()[0].item()
        choice = self.choices[choice_idx]
        print(f'space id: {self.space_id}, current choice: {choice}, '
              f'max choice: {self.choices[-1]}')

        self._dynamic_op.set_choice(choice)

        return self._dynamic_op(x)


class MutableOP(MutableModule):
    """An important type of ``MUTABLES``, inherits from ``MutableModule``.

    Args:
        choices (dict): The configs for the choices, the chosen ``OPS`` used to
            combine ``MUTABLES``.
        choice_args (dict): The args used to set chosen ``OPS``.
    """

    def __init__(self, choices, choice_args, **kwargs):
        super(MutableOP, self).__init__(**kwargs)
        self.choices = self.build_choices(choices, choice_args)
        self.choice_mask = self.build_choice_mask()
        self.full_choice_names = copy.deepcopy(self.choice_names)

    def build_choices(self, cfgs, choice_args):
        """Build all chosen ``OPS`` used to combine ``MUTABLES``, and the
        choices will be sampled.

        Args:
            cfgs (dict): The configs for the choices.
            choice_args (dict): The args used to set chosen ``OPS``.

        Returns:
            torch.nn.ModuleDict: Consists of chosen ``OPS`` in the arg `cfgs`.
        """
        choices = nn.ModuleDict()
        for name, cfg in cfgs.items():
            cfg.update(choice_args)
            op_module = build_op(cfg)
            choices.add_module(name, op_module)
        return choices


@MUTABLES.register_module()
class OneShotOP(MutableOP):
    """A type of ``MUTABLES`` for the one-shot NAS."""

    def __init__(self, **kwargs):
        super(OneShotOP, self).__init__(**kwargs)
        assert self.num_chosen == 1

    def forward(self, x):
        """Forward computation for chosen ``OPS``, in one-shot NAS, the number
        of chosen ``OPS`` can only be one.

        Args:
            x (tensor | tuple[tensor]): x could be a Torch.tensor or a tuple of
                Torch.tensor, containing input data for forward computation.

        Returns:
            torch.Tensor: The result of forward.
        """
        outputs = list()
        for name, chosen_bool in zip(self.full_choice_names, self.choice_mask):
            if name not in self.choice_names:
                continue
            if not chosen_bool:
                continue
            module = self.choices[name]
            outputs.append(module(x))

        assert len(outputs) > 0

        return sum(outputs)


@MUTABLES.register_module()
class DifferentiableOP(MutableOP):
    """Differentiable OP.

    Search the best module from choices by learnable parameters.

    Args:
        with_arch_param (bool): whether build learable architecture parameters.
    """

    def __init__(self, with_arch_param, **kwargs):
        super(DifferentiableOP, self).__init__(**kwargs)
        self.with_arch_param = with_arch_param

    def build_arch_param(self):
        """build learnable architecture parameters."""
        if self.with_arch_param:
            return nn.Parameter(torch.randn(self.num_choices) * 1e-3)
        else:
            return None

    def compute_arch_probs(self, arch_param):
        """compute chosen probs according architecture parameters."""
        return F.softmax(arch_param, -1)

    def forward(self, x, arch_param=None):
        """forward function.

        In some algorithms, there are several ``MutableModule`` share the same
        architecture parameters. So the architecture parameters are passed
        in as args.

        Args:
            prev_inputs (list[torch.Tensor]): each choice's inputs.
            arch_param (torch.nn.Parameter): architecture parameters.
        """
        if self.with_arch_param:
            assert arch_param is not None, \
                f'In {self.space_id}, the arch_param can not be None when the \
                    with_arch_param=True.'

            # 1. compute choices' probs.
            probs = self.compute_arch_probs(arch_param)

            # 2. compute every op's outputs.
            outputs = list()
            for prob, module in zip(probs, self.choice_modules):
                if prob > 0:
                    outputs.append(prob * module(x))

        else:
            outputs = list()
            for name, chosen_bool in zip(self.full_choice_names,
                                         self.choice_mask):
                if name not in self.choice_names:
                    continue
                if not chosen_bool:
                    continue
                module = self.choices[name]
                outputs.append(module(x))

            assert len(outputs) > 0
        return sum(outputs)


@MUTABLES.register_module()
class GumbelOP(DifferentiableOP):
    """Gumbel OP.

    Search the best module from choices by gumbel trick.
    """

    def __init__(self, tau=1.0, hard=True, **kwargs):
        super(GumbelOP, self).__init__(**kwargs)
        self.tau = tau
        self.hard = hard

    def set_temperature(self, temperature):
        """Modify the temperature."""
        self.temperature = temperature

    def compute_arch_probs(self, arch_param):
        """compute chosen probs by gumbel trick."""
        probs = F.gumbel_softmax(
            arch_param, tau=self.tau, hard=self.hard, dim=-1)
        return probs
