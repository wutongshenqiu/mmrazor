# Copyright (c) OpenMMLab. All rights reserved.
from .misc import replace_module
from .switchable_bn import SwitchableBatchNorm2d

__all__ = ['SwitchableBatchNorm2d', 'replace_module']
