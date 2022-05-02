# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional

import mmcv
from mmcv.cnn import MODELS
from mmcv.runner import BaseModule
from torch.nn import Module

from mmrazor.models.mutables.base import BaseMutable
from mmrazor.utils import master_only_print


class BaseArchitecture(BaseModule):
    """Base class for architecture.

    Args:
        model (:obj:`torch.nn.Module`): Model to be slimmed, such as
            ``DETECTOR`` in MMDetection.
    """

    def __init__(self,
                 model_cfg: Dict,
                 mutable_cfg_path: Optional[str] = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)

        self.model: Module = MODELS.build(model_cfg)
        master_only_print(self.model)

        # TODO
        # pruner subnet deploy
        if mutable_cfg_path is not None:
            mutable_cfg = mmcv.fileio.load(mutable_cfg_path)
            self.deploy_mutable_subnet(mutable_cfg)

    def deploy_mutable_subnet(self, mutable_cfg: Dict) -> None:
        for name, module in self.model.named_modules():
            if isinstance(module, BaseMutable):
                subnet_config = mutable_cfg[name]
                module.deploy_subnet(subnet_config)
                master_only_print(f'module: {name} deploy with subnet config: '
                                  f'{subnet_config}')

    def export_mutable_subnet(self) -> Dict:
        mutable_cfg = dict()
        for name, module in self.model.named_modules():
            if isinstance(module, BaseMutable):
                mutable_cfg[name] = module.export_subnet()

        return mutable_cfg

    def forward_dummy(self, img):
        """Used for calculating network flops."""
        assert hasattr(self.model, 'forward_dummy')
        return self.model.forward_dummy(img)

    def forward(self, img, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.

        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``resturn_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        return self.model(img, return_loss=return_loss, **kwargs)

    def simple_test(self, img, img_metas):
        """Test without augmentation."""
        return self.model.simple_test(img, img_metas)

    def show_result(self, img, result, **kwargs):
        """Draw `result` over `img`"""
        return self.model.show_result(img, result, **kwargs)
