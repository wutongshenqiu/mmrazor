# Copyright (c) OpenMMLab. All rights reserved.
import copy
from types import MethodType
from typing import Any, Callable, Dict, Hashable, Iterable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import get_model_complexity_info

# TODO: import to __init__.py
from mmrazor.models.architectures.base import BaseArchitecture
from mmrazor.models.builder import PRUNERS
from .structure_pruning import StructurePruner
from .types import METRIC_DICT_TYPE, SUBNET_TYPE
from .utils import get_module, replace_module


def _print_debug_msg(*args, rank_idx: int = 0, **kwargs) -> None:
    try:
        if torch.distributed.get_rank() == rank_idx:
            print(f'rank {rank_idx}', end=', ')
            print(*args, **kwargs)
    except RuntimeError:
        print(*args, **kwargs)


class CompactorLayer(nn.Module):
    # TODO: update docs

    def __init__(self, feature_nums: int, name: str) -> None:
        super().__init__()

        self._name = name
        self._layer = nn.Conv2d(
            in_channels=feature_nums,
            out_channels=feature_nums,
            kernel_size=(1, 1),
            bias=False)
        with torch.no_grad():
            self._layer.weight.data.copy_(
                torch.eye(feature_nums).reshape(self._layer.weight.shape))

        # TODO
        # will this affect other algorithms?
        self.register_buffer(
            name='mask',
            tensor=self._layer.weight.new_ones((feature_nums, 1, 1, 1)))

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
        lasso_grad = self.lasso_grad(lasso_strength)
        self._layer.weight.grad.data.add_(lasso_grad)

    @torch.no_grad()
    def get_metric_list(self) -> List[float]:
        return self.get_metric_tensor().tolist()

    @torch.no_grad()
    def get_metric_tensor(self) -> torch.Tensor:
        return torch.sqrt(torch.sum(self._layer.weight**2, dim=(1, 2, 3)))

    @torch.no_grad()
    def filter_ids_ge_threshold(self, threshold: float) -> torch.Tensor:
        metric_tensor = self.get_metric_tensor()
        return torch.where(metric_tensor >= threshold)[0]

    @torch.no_grad()
    def filter_ids_l_threshold(self, threshold: float) -> torch.Tensor:
        metric_tensor = self.get_metric_tensor()
        return torch.where(metric_tensor < threshold)[0]

    @property
    def deactivated_filter_nums(self) -> int:
        return (self.mask == 0.).sum().item()

    @property
    def name(self) -> str:
        return self._name

    @property
    def weight(self) -> torch.Tensor:
        return self._layer.weight


@PRUNERS.register_module()
class ResRepPruner(StructurePruner):
    # TODO: update docs
    def __init__(self,
                 flops_constraint: Optional[float] = None,
                 flops_ratio: Optional[float] = None,
                 begin_granularity: int = 4,
                 ignore_skip_mask: bool = True,
                 least_channel_nums: int = 1,
                 lasso_strength: float = 1e-4,
                 input_shape: Iterable[int] = (3, 224, 224),
                 follow_paper: bool = True,
                 threshold: float = 1e-5,
                 **kwargs: Any) -> None:
        if (flops_constraint is None and flops_ratio is None) or \
                (flops_constraint is not None and flops_ratio is not None):
            raise ValueError(
                'One and only one of `flops_constraint` or `flops_ratio` '
                'must be specified')
        if flops_ratio is not None and (flops_ratio <= 0 or flops_ratio > 1):
            raise ValueError(
                f'`flops_ratio` must between 0 and 1, but got `{flops_ratio}`')
        if least_channel_nums < 1:
            raise ValueError(f'`least_channel_nums` must greater than 0, '
                             f'but got `{least_channel_nums}`')
        if begin_granularity < 1:
            raise ValueError(f'`begin_granularity` must greater than 0, '
                             f'but got `{begin_granularity}`')
        if lasso_strength <= 0:
            raise ValueError(f'`lasso_strength` must greater than 0, '
                             f'but got `{lasso_strength}`')
        if threshold < 0:
            raise ValueError(f'`threshold` must greater than or equal to 0, '
                             f'but got `{threshold}`')

        super(ResRepPruner, self).__init__(**kwargs)

        self._flops_constraint = flops_constraint
        self._flops_ratio = flops_ratio
        self._begin_granularity = begin_granularity
        self._ignore_skip_mask = ignore_skip_mask
        if not ignore_skip_mask:
            self._pre_deactivated_nums = 0
        self._least_channel_nums = least_channel_nums
        self._lasso_strength = lasso_strength
        self._input_shape = input_shape
        self._follow_paper = follow_paper
        self._threshold = threshold

    # TODO
    # should other layer except compactor has mask?
    def prepare_from_supernet(self, supernet: BaseArchitecture) -> None:
        self._init_flops(supernet)
        super().prepare_from_supernet(supernet)

        group2modules = dict()
        for module_name, group in self.module2group.items():
            try:
                group2modules[group].append(module_name)
            except KeyError:
                group2modules[group] = [module_name]

        # HACK: just for verify correction
        self._conv_norm_links = {v: k for k, v in self.norm_conv_links.items()}

        compactor2modules: Dict[str, List[str]] = dict()
        compactors = nn.ModuleDict()
        for name, out_mask in self.channel_spaces.items():
            if name in group2modules:
                modules_name = group2modules[name]
            else:
                modules_name = [name]
            if self._follow_paper:
                if len(modules_name) != 1:
                    continue
                # HACK: very ugly hack
                if name == 'backbone.conv1':
                    continue

            out_channels = out_mask.size(1)
            # nn.ModuleDict's key should not contain '.'
            compactor_name = name.replace('.', '_')
            compactors[compactor_name] = CompactorLayer(
                feature_nums=out_channels, name=compactor_name)

            for module_name in modules_name:
                if module_name not in self._conv_norm_links:
                    _print_debug_msg(
                        f'{module_name} does not have a bn layer follow, skip!'
                    )
                    continue
                module = self.name2module[module_name]
                if isinstance(module, nn.Conv2d) and module.groups == 1:
                    follow_norm_name = self._conv_norm_links[module_name]
                    follow_norm_module = self.name2module[follow_norm_name]
                    if not isinstance(follow_norm_module, nn.BatchNorm2d):
                        continue

                    module.__compactor_name__ = compactor_name
                    follow_norm_module.__original_forward = \
                        follow_norm_module.forward
                    follow_norm_module.forward = self.modify_bn_forward(
                        module=follow_norm_module,
                        compactor=compactors[compactor_name])
                    _print_debug_msg(
                        f'modify {follow_norm_name} with compactor: '
                        f'{compactor_name}')
                    # original code
                    # module.__compactor_name__ = compactor_name
                    # module.forward = self.modify_conv_forward(
                    #     module=module, compactor=compactors[compactor_name])
                try:
                    compactor2modules[compactor_name].append(module_name)
                except KeyError:
                    compactor2modules[compactor_name] = [module_name]
        self._compactors: Dict[Hashable, CompactorLayer] = compactors
        self._module2compactor = self._map_conv_compactor()
        self._compactor2modules = compactor2modules

        _print_debug_msg(f'module to compactors: {self._module2compactor}')
        compactor2modules_msg = f'total compactors: {len(compactor2modules)}\n'
        compactor2modules_msg += '\n'.join(
            (f'compactor name: {cn}, mask shape: {c.mask.shape}, '
             f'corresponding modules: {compactor2modules[cn]}'
             for cn, c in compactors.items()))
        _print_debug_msg(compactor2modules_msg)

        # HACK: to align with checkpoint
        self._set_convs_mask(self.sample_subnet(supernet))

    @staticmethod
    def modify_bn_forward(module,
                          compactor: CompactorLayer) -> Callable[..., Any]:
        """Modify bn layer's forward method, add the operation of compactor
        after convolution.

        Args:
            module ([type]): [description]
            compactor (CompactorLayer): [description]

        Returns:
            Callable[..., Any]: [description]
        """

        def modified_forward(self, input: torch.Tensor) -> torch.Tensor:
            out = self.__original_forward(input)
            out = compactor(out)

            return out

        return MethodType(modified_forward, module)

    @staticmethod
    def modify_conv_forward(module,
                            compactor: CompactorLayer) -> Callable[..., Any]:
        """Modify convolutional layer's forward method, add the operation of
        compactor after convolution.

        Args:
            module ([type]): [description]
            compactor (CompactorLayer): [description]

        Returns:
            Callable[..., Any]: [description]
        """

        def modified_forward(self, feature: torch.Tensor) -> torch.Tensor:
            out = F.conv2d(feature, self.weight, self.bias, self.stride,
                           self.padding, self.dilation, self.groups)
            out = compactor(out)

            return out

        return MethodType(modified_forward, module)

    def add_pruning_attrs(self,
                          module: nn.Module,
                          is_modify_forward: bool = False) -> None:
        """Override method in parent class, change default value of
        `is_modify_forward` to False.

        Args:
            module (nn.Module): [description]
            is_modify_forward (bool, optional): [description]
        """
        return super().add_pruning_attrs(module, is_modify_forward)

    def update_mask(self, supernet: BaseArchitecture) -> None:
        self.set_subnet(self.sample_subnet(supernet))

    def gradient_reset(self) -> None:
        for compactor in self._compactors.values():
            compactor.add_lasso_grad(self._lasso_strength)

    # TODO
    # use `sample_subnet` may ruin function signature
    def sample_subnet(self, supernet: BaseArchitecture) -> SUBNET_TYPE:
        return self._calc_compactors_mask(supernet)

    @torch.no_grad()
    def _calc_compactors_mask(self, supernet: BaseArchitecture) -> SUBNET_TYPE:
        """Calculating mask for each compactor according to compactors' metric.

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
            if self._ignore_skip_mask:
                next_deactivated_nums = self._get_deactivated_filter_nums(
                    compactors_mask=cur_compactors_mask) + \
                    self._begin_granularity
            else:
                next_deactivated_nums = self._pre_deactivated_nums + \
                    self._begin_granularity
                self._pre_deactivated_nums = next_deactivated_nums
        else:
            next_deactivated_nums = float('inf')
        assert next_deactivated_nums > 0
        _print_debug_msg(
            f'next total deactivated numbers: {next_deactivated_nums}')

        next_compactors_mask = {
            k: torch.ones_like(v.mask)
            for k, v in self._compactors.items()
        }
        cur_deactivated_nums = 0
        skip_deactivated_nums = 0
        while True:
            cur_deactivated_nums += 1
            compactor_name, filter_id = sorted_metric_keys[
                cur_deactivated_nums]
            compactor_mask = next_compactors_mask[compactor_name]
            if self._get_mask_activated_filter_nums(compactor_mask) <= \
                    self._least_channel_nums:
                skip_deactivated_nums += 1
                continue
            self._set_deactivated_filter(compactor_mask, filter_id)

            cur_flops = self._calc_subnet_flops(
                supernet=supernet, compactors_mask=next_compactors_mask)
            if cur_flops <= self._flops_constraint:
                break

            if cur_deactivated_nums >= next_deactivated_nums:
                break

        _print_debug_msg(f'skip deactivated numbers: {skip_deactivated_nums}')
        _print_debug_msg(
            f'current FLOPs rate: {cur_flops / self._original_flops}')
        compactor_mask_info = '\n'.join(
            (f'{k}, deactivated numbers: '
             f'{self._get_mask_deactivated_filter_nums(v)}, '
             f'activated numbers: {self._get_mask_activated_filter_nums(v)}, '
             f'total numbers: {v.shape[0]}'
             for k, v in next_compactors_mask.items()))
        _print_debug_msg(compactor_mask_info)
        return next_compactors_mask

    # TODO
    # redundant function as in `AutoSlim`
    # put here or in algorithm/resrep.py?
    def _init_flops(self, supernet: BaseArchitecture) -> None:
        """Add `__flops__` to each module in supernet.

        Args:
            supernet (BaseArchitecture): [description]
        """
        flops_model = copy.deepcopy(supernet)
        flops_model.eval()
        if hasattr(flops_model, 'forward_dummy'):
            flops_model.forward = flops_model.forward_dummy
        else:
            raise NotImplementedError(
                'FLOPs counter is currently not currently supported with {}'.
                format(flops_model.__class__.__name__))

        flops, _ = get_model_complexity_info(
            flops_model,
            self._input_shape,
            print_per_layer_stat=False,
            as_strings=False)
        self._original_flops: float = flops
        if self._flops_ratio:
            self._flops_constraint = flops * self._flops_ratio

        flops_lookup = dict()
        for name, module in flops_model.named_modules():
            flops = getattr(module, '__flops__', 0)
            flops_lookup[name] = flops
        del (flops_model)

        for name, module in supernet.named_modules():
            module.__flops__ = flops_lookup[name]

    # TODO
    # may not easy to implement
    def _calc_subnet_flops(self, supernet: BaseArchitecture,
                           compactors_mask: SUBNET_TYPE) -> float:
        """Calculate subnet flops.

        Args:
            supernet (BaseArchitecture): [description]
            compactors (SUBNET_TYPE): [description]

        Returns:
            float: [description]
        """
        flops = 0
        for name, module in supernet.named_modules():
            module_name = name.replace('model.', '', 1)
            if module_name in self._module2compactor['out_mask']:
                compactor_name = self._module2compactor['out_mask'][
                    module_name]
                compactor_mask = compactors_mask[compactor_name]
                out_mask_ratio = float(compactor_mask.sum() /
                                       compactor_mask.numel())
            else:
                out_mask_ratio = 1
            if module_name in self._module2compactor['in_mask']:
                compactor_name = self._module2compactor['in_mask'][module_name]
                compactor_mask = compactors_mask[compactor_name]
                in_mask_ratio = float(compactor_mask.sum() /
                                      compactor_mask.numel())
            else:
                in_mask_ratio = 1

            flops += module.__flops__ * out_mask_ratio * in_mask_ratio

        return flops

    def _get_deactivated_filter_nums(self,
                                     compactors_mask: SUBNET_TYPE) -> int:
        """Calculate deactivated filter numbers of all compactors.

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
        return (compactor_mask == 0.).sum().item()

    @staticmethod
    def _get_mask_activated_filter_nums(comapctor_mask: torch.Tensor) -> int:
        """Calculate activated filter numbers of a compactor mask.

        Args:
            comapctor_mask (torch.Tensor): [description]

        Returns:
            int: [description]
        """
        return int(comapctor_mask.sum().item())

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
        compactor_mask[filter_id, :, :, :] = 0.0

    def _get_metric_dict(self) -> METRIC_DICT_TYPE:
        """Calculate metric for every output channel of each compactor.

        Returns:
            METRIC_DICT_TYPE: (compactor_name, filter_id) -> metric
        """
        metric_dict = dict()

        for compactor_name, compactor in self._compactors.items():
            metric_list = compactor.get_metric_list()
            for i in range(len(metric_list)):
                metric_dict[(compactor_name, i)] = metric_list[i]

        return metric_dict

    @torch.no_grad()
    def set_subnet(self, subnet_dict: SUBNET_TYPE) -> None:
        """Modify mask of compactor and module in supernet according to
        subnet_dict.

        Changes of `in_mask` and `out_mask` for conv in supernet are just for
        compatibility of `StructPruner`, since they are not actually used in
        `forward` part, remember that we only prune compactor during training.

        Args:
            subnet_dict (SUBNET_TYPE): [description]
        """
        self._set_convs_mask(subnet_dict)

        for compactor_name, compactor in self._compactors.items():
            compactor.mask = \
                subnet_dict[compactor_name].to(compactor.mask.device)

    @torch.no_grad()
    def _set_convs_mask(self, subnet_dict: SUBNET_TYPE) -> None:
        """Modify mask of convolutional layer according to compactors' mask.

        Since we only consider the pruning of `Conv2d`, the function may be
        different to `set_mask`

        Args:
            subnet_dict (SUBNET_TYPE): [description]
        """
        for module_name, compactor_name in \
                self._module2compactor['out_mask'].items():
            module = self.name2module[module_name]
            module.out_mask = \
                subnet_dict[compactor_name].to(module.out_mask.device)

        for module_name, compactor_name in \
                self._module2compactor['in_mask'].items():
            module = self.name2module[module_name]
            module.in_mask = \
                subnet_dict[compactor_name].to(module.in_mask.device)

    def _map_conv_compactor(self) -> Dict[str, Dict[str, str]]:
        """Map `in_mask` and `out_mask` of each convolutional layer to its
        correspongding compactor.

        Returns:
            Dict[str, Dict[str, str]]: [description]
        """
        module2compactor = dict()

        for module_name in self.modules_have_child:
            module = self.name2module[module_name]
            if not hasattr(module, '__compactor_name__'):
                continue
            compactor_name = getattr(module, '__compactor_name__')
            try:
                module2compactor['out_mask'][module_name] = compactor_name
            except KeyError:
                module2compactor['out_mask'] = {module_name: compactor_name}

        for module_name in self.modules_have_ancest:
            parents = self.node2parents[module_name]
            parent = parents[0]
            parent_module = self.name2module[parent]
            if not hasattr(parent_module, '__compactor_name__'):
                continue
            compactor_name = getattr(parent_module, '__compactor_name__')
            try:
                module2compactor['in_mask'][module_name] = compactor_name
            except KeyError:
                module2compactor['in_mask'] = {module_name: compactor_name}

        return module2compactor

    def set_min_channel(self) -> None:
        pass

    # TODO
    # should record which channel have been set to 0
    def export_subnet(self) -> Dict[Hashable, Any]:
        return super().export_subnet()

    @torch.no_grad()
    def deploy_subnet(self, supernet: BaseArchitecture,
                      channel_cfg: Dict[Hashable, Any]) -> None:
        model = supernet.model

        identity_layer = channel_cfg.pop('identity_layers', None)
        if identity_layer is not None:
            for layer_name in identity_layer:
                replace_module(model, layer_name, nn.Identity())
                _print_debug_msg(f'replace `{layer_name}` with Identity.')

        for conv_name, cfg in channel_cfg.items():
            conv_module = get_module(model, conv_name)
            assert isinstance(conv_module, nn.Conv2d)
            weight = conv_module.weight

            out_channels = cfg.get('out_channels')
            if out_channels is not None:
                raw_out_channels = cfg.get('raw_out_channels')
                assert weight.shape[0] == raw_out_channels
                weight = weight[:out_channels, :, :, :]
                conv_module.out_channels = out_channels
                conv_module.bias = nn.Parameter(torch.zeros(out_channels))
                _print_debug_msg(
                    f'Slice out channels of {conv_name} to {out_channels}')
            in_channels = cfg.get('in_channels')
            if in_channels is not None:
                raw_in_channels = cfg.get('raw_in_channels')
                assert weight.shape[1] == raw_in_channels
                weight = weight[:, :in_channels, :, :]
                conv_module.in_channels = in_channels
                _print_debug_msg(
                    f'Slice in channels of {conv_name} to {in_channels}')
            conv_module.weight = nn.Parameter(weight)

        delattr(self, '_compactors')

    # TODO: ugly hack
    # DDP bug
    def forward(self, architecture: BaseArchitecture, data: Dict) -> Dict:
        return architecture(**data)
