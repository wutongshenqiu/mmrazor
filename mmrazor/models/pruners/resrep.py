# Copyright (c) OpenMMLab. All rights reserved.
from types import MethodType
from typing import Any, Callable, Dict, Hashable, List

import torch
import torch.nn as nn
import torch.nn.functional as F

# TODO: import to __init__.py
from mmrazor.models.architectures.base import BaseArchitecture
from mmrazor.models.builder import PRUNERS
from .structure_pruning import StructurePruner
from .types import METRIC_DICT_TYPE, SUBNET_TYPE


class CompactorLayer(nn.Module):
    # TODO: update docs

    def __init__(self, feature_nums: int) -> None:
        super().__init__()

        self._layer = nn.Conv2d(
            in_channels=feature_nums,
            out_channels=feature_nums,
            kernel_size=(1, 1),
            bias=False)
        with torch.no_grad():
            self._layer.weight.data.copy_(
                torch.eye(feature_nums).reshape(self._layer.weight.shape))

        self.register_buffer(
            name='mask',
            tensor=self._layer.weight.new_ones((1, feature_nums, 1, 1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._layer(x)

    @torch.no_grad()
    def lasso_grad(self, lasso_strength: float) -> torch.Tensor:
        weight = self._layer.weight.data
        grad = self._layer.weight.grad
        mask = self.mask

        grad.data *= mask
        lasso_grad = weight * \
            ((weight ** 2).sum(dim=(1, 2, 3), keepdim=True) ** (-0.5))
        return lasso_strength * lasso_grad

    @torch.no_grad()
    def add_lasso_grad(self, lasso_strength: float) -> None:
        self._layer.weight.grad.add_(self.lasso_grad(lasso_strength))

    @torch.no_grad()
    def get_metric_list(self) -> List[float]:
        return torch.sqrt(torch.sum(self._layer.weight**2,
                                    dim=(1, 2, 3))).tolist()

    @property
    def deactivated_filter_nums(self) -> int:
        return (self.mask == 0.0).sum().item()


@PRUNERS.register_module()
class ResRepPruner(StructurePruner):
    # TODO: update docs
    def __init__(self,
                 flops_constraint: float,
                 begin_granularity: int = 4,
                 least_channel_nums: int = 1,
                 lasso_strength: float = 1e-4,
                 **kwargs: Any) -> None:
        super(ResRepPruner, self).__init__(**kwargs)

        self._flops_constraint = flops_constraint
        self._begin_granularity = begin_granularity
        self._least_channel_nums = least_channel_nums
        self._lasso_strength = lasso_strength

    # TODO
    # should other layer except compactor has mask?
    def prepare_from_supernet(self, supernet: BaseArchitecture) -> None:
        super().prepare_from_supernet(supernet)

        group2module = dict()
        for module_name, group in self.module2group.items():
            module = self.name2module[module_name]
            try:
                group2module[group].append(module)
            except KeyError:
                group2module[group] = [module]

        compactors = nn.ModuleDict()
        for name, out_mask in self.channel_spaces.items():
            out_channels = out_mask.size(1)
            # nn.ModuleDict's key should not contain '.'
            compactor_name = name.replace('.', '_')
            compactors[compactor_name] = CompactorLayer(out_channels)

            if name in group2module:
                modules = group2module[name]
            else:
                modules = [self.name2module[name]]

            for module in modules:
                module._pruning_group = compactor_name
                if type(module).__name__ == 'Conv2d' and module.groups == 1:
                    module.forward = self.modify_conv_forward(
                        module=module, compactors=compactors)

        self._compactors: Dict[Hashable, CompactorLayer] = compactors

    # TODO
    # pass single compactor instead of compactors
    @staticmethod
    def modify_conv_forward(
            module, compactors: Dict[str, nn.Module]) -> Callable[..., Any]:
        """Modify the forward method of a conv layer."""

        def modified_forward(self, feature: torch.Tensor) -> torch.Tensor:
            out = F.conv2d(feature, self.weight, self.bias, self.stride,
                           self.padding, self.dilation, self.groups)
            compactor = compactors[self._pruning_group]
            out = compactor(out)

            return out

        return MethodType(modified_forward, module)

    def add_pruning_attrs(self,
                          module: nn.Module,
                          is_modify_forward: bool = False) -> None:
        return super().add_pruning_attrs(module, is_modify_forward)

    def update_mask(self, supernet: BaseArchitecture) -> None:
        self.set_subnet(self._calc_compactors_mask(supernet))

    def gradient_reset(self) -> None:
        for compactor in self._compactors.values():
            compactor.add_lasso_grad(self._lasso_strength)

    # TODO
    # use `sample_subnet` may ruin function signature
    def sample_subnet(self) -> SUBNET_TYPE:
        pass

    @torch.no_grad()
    def _calc_compactors_mask(self, supernet: BaseArchitecture) -> SUBNET_TYPE:
        """Calculating mask for each compactor.

        Args:
            supernet (BaseArchitecture): [description]

        Returns:
            SUBNET_TYPE: [description]
        """
        metric_dict = self._get_metric_dict()
        sorted_metric_keys = sorted(metric_dict, key=metric_dict.get)

        cur_compactors_mask = {
            k: v.mask.clone()
            for k, v in self._compactors.items()
        }
        cur_flops = self._calc_subnet_flops(
            supernet=supernet, compactors_mask=cur_compactors_mask)
        if cur_flops > self._flops_constraint:
            next_deactivated_nums = self._get_deactivated_filter_nums(
                compactors_mask=cur_compactors_mask) + self._begin_granularity
        else:
            next_deactivated_nums = float('inf')
        assert next_deactivated_nums > 0

        next_compactors_mask = {
            k: torch.ones_like(v.mask)
            for k, v in self._compactors.items()
        }
        cur_deactivated_nums = 0
        while True:
            compactor_name, filter_id = sorted_metric_keys[
                cur_deactivated_nums]
            compactor_mask = next_compactors_mask[compactor_name]
            if self._get_mask_deactivated_filter_nums(compactor_mask) < \
                    self._least_channel_nums:
                cur_deactivated_nums += 1
                continue
            cur_deactivated_nums += 1
            self._set_deactivated_filter(compactor_mask, filter_id)

            cur_flops = self._calc_subnet_flops(
                supernet=supernet, compactors_mask=next_compactors_mask)
            if cur_flops <= self._flops_constraint:
                break

            if cur_deactivated_nums >= next_deactivated_nums:
                break

        return next_compactors_mask

    # TODO
    # may not easy to implement
    @staticmethod
    def _calc_subnet_flops(supernet: BaseArchitecture,
                           compactors_mask: SUBNET_TYPE) -> float:
        """Calculate subnet flops.

        Args:
            supernet (BaseArchitecture): [description]
            compactors (SUBNET_TYPE): [description]

        Returns:
            float: [description]
        """
        raise NotImplementedError

    def _get_deactivated_filter_nums(self,
                                     compactors_mask: SUBNET_TYPE) -> int:
        """Calculate deactivated filter numbers of compactors.

        Args:
            compactors_mask (SUBNET_TYPE): [description]

        Returns:
            int: [description]
        """
        return sum((self._get_mask_deactivated_filter_nums(mask)
                    for mask in compactors_mask.values()))

    @staticmethod
    def _get_mask_deactivated_filter_nums(compactor_mask: torch.Tensor) -> int:
        """Calculate deactivated filter numbers of a compactor mask.

        Args:
            compactor_mask (torch.Tensor): [description]

        Returns:
            int: [description]
        """
        return (compactor_mask == 0.0).sum().item()

    @staticmethod
    def _set_deactivated_filter(compactor_mask: torch.Tensor,
                                filter_id: int) -> int:
        """Set deactivated filter of a compactor mask according to filter_id.

        Args:
            compactor_mask (torch.Tensor): [description]
            filter_id (int): [description]

        Returns:
            int: [description]
        """
        compactor_mask[:, filter_id, :, :] = 0.0

    def _get_metric_dict(self) -> METRIC_DICT_TYPE:
        """Calculate metric for every output channel of each compactor.

        Returns:
            METRIC_DICT_TYPE: (compactor_name, filter_id) -> metric
        """
        metric_dict = dict()

        for compactor_name, compactor in self._compactors.items():
            metric_list = compactor.get_metric_list()
            for i in range(len(metric_list)):
                metric_list[(compactor_name, i)] = metric_list[i]

        return metric_dict

    @torch.no_grad()
    def set_subnet(self, subnet_dict: SUBNET_TYPE) -> None:
        """Modify compactors' mask according to subnet_dict.

        Args:
            subnet_dict (SUBNET_TYPE): [description]
        """
        for compactor_name, compactor in self._compactors.items():
            compactor.mask = \
                subnet_dict[compactor_name].to(compactor.mask.device)

    def set_min_channel(self) -> None:
        pass
