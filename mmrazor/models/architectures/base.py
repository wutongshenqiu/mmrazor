# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict

import mmcv
from mmcv.cnn import MODELS
from mmcv.runner import BaseModule
from torch.nn import Module

from mmrazor.models.mutables.mixins import MutableMixin


class BaseArchitecture(BaseModule):
    """Base class for architecture.

    Args:
        model (:obj:`torch.nn.Module`): Model to be slimmed, such as
            ``DETECTOR`` in MMDetection.
    """

    def __init__(self, model, mutable_cfg=None, channel_cfg=None, **kwargs):
        super(BaseArchitecture, self).__init__(**kwargs)
        self.model = MODELS.build(model)
        if mutable_cfg:

            mutable_cfg = mmcv.fileio.load(mutable_cfg)
            self.deploy_subnet_mutables(self.model, mutable_cfg)
            print(self.model)

        # TODO support channel cfg

    def deploy_subnet_mutables(self, supernet: Module,
                               subnet_dict: Dict[str, Dict]) -> None:
        """Export the subnet from the supernet based on the specified
        subnet_dict.

        Args:
            supernet (:obj:`torch.nn.Module`): The architecture to be used in
                your algorithm.
            subnet_dict (dict): Record the information to build the subnet from
                the supernet,
                its keys are the properties ``space_id`` of placeholders in the
                mutator's search spaces,
                its values are dicts: {'chosen': ['chosen name1',
                'chosen name2', ...]}
        """

        for name, module in supernet.named_modules():
            if isinstance(module, MutableMixin):
                chosen = subnet_dict[name]
                module.deploy(chosen)

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
