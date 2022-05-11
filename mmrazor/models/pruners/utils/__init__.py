# Copyright (c) OpenMMLab. All rights reserved.
from .misc import get_module, replace_module
from .switchable_bn import SwitchableBatchNorm2d

__all__ = ['SwitchableBatchNorm2d', 'replace_module', 'get_module']
