# Copyright (c) OpenMMLab. All rights reserved.
from functools import reduce

from torch import nn


def _get_parent_module_by_name(model: nn.Module,
                               module_name: str) -> nn.Module:
    module_names = module_name.split('.')
    if len(module_names) == 1:
        return model
    parent_module_names = module_names[:-1]

    return reduce(getattr, parent_module_names, model)


def replace_module(model: nn.Module, module_name: str,
                   new_module: nn.Module) -> None:
    parent_module = _get_parent_module_by_name(model, module_name)
    child_module_name = module_name.split('.')[-1]

    setattr(parent_module, child_module_name, new_module)
