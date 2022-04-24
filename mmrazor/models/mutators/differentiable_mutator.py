# Copyright (c) OpenMMLab. All rights reserved.
from mmrazor.models.builder import MUTATORS
from .base import BaseMutator


@MUTATORS.register_module()
class DifferentiableMutator(BaseMutator):
    """A mutator for the differentiable NAS, which mainly provide some core
    functions of changing the structure of ``ARCHITECTURES``."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
