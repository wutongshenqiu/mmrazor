# Copyright (c) OpenMMLab. All rights reserved.
import random
from typing import Any, Dict, List

import torch
from pydantic import BaseModel

from mmrazor.models.architectures.base import BaseArchitecture
from mmrazor.models.builder import PRUNERS
from mmrazor.utils import master_only_print
from .structure_pruning import StructurePruner


class _SpecialConfig(BaseModel):
    in_key: str
    refer: str
    expand_ratio: int


class _RangeConfig(BaseModel):
    start_key: str
    min_channels: int
    priority: int = 1
    specials: List[_SpecialConfig] = []


@PRUNERS.register_module()
class RangePruner(StructurePruner):

    def __init__(self, range_config: Dict[str, Dict[str, Any]],
                 **kwargs) -> None:
        super().__init__(**kwargs)

        self._name2config = {
            k: _RangeConfig(**v)
            for k, v in range_config.items()
        }

    def prepare_from_supernet(self, supernet: BaseArchitecture) -> None:
        super().prepare_from_supernet(supernet)

        self._space2modules = self._map_space_to_modules()
        self._space2config = self._map_space_to_config()
        self._module2space = {
            module: space
            for space, modules in self._space2modules.items()
            for module in modules
        }

        # HACK
        # channel width of some modules will
        # depend on their parent's channel width,
        # so channel width mask must be sampled by order.
        # This also require the order of the same config,
        # in current situation, it seems ok.
        ordered_config_list = list(self._name2config.keys())
        sorted_spaces = sorted(
            self.channel_spaces.keys(),
            key=lambda x: ordered_config_list.index(self._space2config[x]))
        self._sorted_spaces = sorted_spaces

        for space_id in self._sorted_spaces:
            master_only_print(
                f'space: {space_id}, config: {self._space2config[space_id]}')
            master_only_print(f'modules: {self._space2modules[space_id]}')

    def _map_space_to_modules(self) -> Dict[str, List[str]]:
        group2modules: Dict[str, List[str]] = dict()
        for name in self.name2module.keys():
            if name in self.module2group:
                group = self.module2group[name]
                try:
                    group2modules[group].append(name)
                except KeyError:
                    group2modules[group] = [name]

        space2modules: Dict[str, List[str]] = dict()
        for space in self.channel_spaces.keys():
            if space in group2modules:
                space2modules[space] = group2modules[space].copy()
            else:
                space2modules[space] = [space]

        return space2modules

    def _map_space_to_config(self) -> Dict[str, str]:
        name2config = self._name2config
        space2modules = self._space2modules

        space2config = dict()
        for space, modules in space2modules.items():
            chosen_config = None
            priority = float('-inf')

            for module in modules:
                for name, config in name2config.items():
                    if module.startswith(config.start_key):
                        if priority < config.priority:
                            priority = config.priority
                            chosen_config = name
                            break

            space2config[space] = chosen_config

        return space2config

    @staticmethod
    def _get_mask_channels(t: torch.Tensor) -> int:
        return t.sum().long().item()

    def sample_subnet(self,
                      sample_policy: str = 'random'
                      ) -> Dict[str, torch.Tensor]:
        valid_sample_policies = {'min', 'max', 'random'}
        if sample_policy not in valid_sample_policies:
            raise ValueError(f'Invalid sample_policy: {sample_policy}, '
                             f'expect one of {valid_sample_policies}')

        config2channels = dict()
        subnet_dict: Dict[str, torch.Tensor] = dict()
        for space_id in self._sorted_spaces:
            old_out_mask = self.channel_spaces[space_id]

            range_config_name = self._space2config[space_id]
            range_config = self._name2config[range_config_name]
            new_channels = None
            for module in self._space2modules[space_id]:
                for special in range_config.specials:
                    if special.in_key in module:
                        master_only_print(
                            f'special module: {module}, '
                            f'parent: {self.node2parents[module][0]}')
                        if special.refer == 'parent':
                            parent_module = self.node2parents[module][0]
                            parent_space = self._module2space[parent_module]
                            parent_out_mask = subnet_dict[parent_space]
                            parent_out_channels = \
                                self._get_mask_channels(parent_out_mask)
                            new_channels = \
                                parent_out_channels * special.expand_ratio
                            break
                        else:
                            raise NotImplementedError(
                                'Only support `parent` refer now, '
                                f'but got: `{special.refer}`!')
                if new_channels is not None:
                    break

            if new_channels is None:
                if range_config_name in config2channels:
                    new_channels = config2channels[range_config_name]
                else:
                    min_channels = range_config.min_channels
                    max_channels = old_out_mask.size(1)

                    if sample_policy == 'min':
                        new_channels = min_channels
                    elif sample_policy == 'max':
                        new_channels = max_channels
                    else:
                        new_channels = random.randint(min_channels,
                                                      max_channels)
                    config2channels[range_config_name] = new_channels

            new_out_mask = torch.zeros_like(old_out_mask)
            new_out_mask[:, :new_channels] = 1

            subnet_dict[space_id] = new_out_mask

        for space_id in self._sorted_spaces:
            mask = subnet_dict[space_id]
            master_only_print('=' * 100)
            master_only_print(
                f'space: {space_id}, layer: {self._space2config[space_id]}')
            for module in self._space2modules[space_id]:
                master_only_print(
                    f'module: {module}, '
                    f'out channels: {self._get_mask_channels(mask)}, '
                    f'max channels: {mask.size(1)}')
            master_only_print('=' * 100)

        return subnet_dict

    def _sample_set_subnet(self, sample_policy: str) -> None:
        subnet_dict = self.sample_subnet(sample_policy)
        self.set_subnet(subnet_dict)

    def set_min_channel(self) -> None:
        self._sample_set_subnet('min')

    def set_max_channel(self) -> None:
        self._sample_set_subnet('max')

    def set_random_channel(self) -> None:
        self._sample_set_subnet('random')
