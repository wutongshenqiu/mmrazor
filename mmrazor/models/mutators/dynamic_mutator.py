# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Dict, List, Optional

from mmrazor.models.architectures.base import BaseArchitecture
from mmrazor.models.builder import MUTATORS
from mmrazor.models.mutables.base import DynamicMutable
from .base import BaseMutator


@MUTATORS.register_module()
class DynamicMutator(BaseMutator):

    def __init__(self,
                 search_groups: Optional[Dict[str, Dict]] = None,
                 **kwargs: Any) -> None:
        super().__init__(**kwargs)

        if search_groups is None:
            search_groups = dict()
        module2group_id = dict()
        for idx, group in enumerate(search_groups):
            for module_name in group['modules']:
                module2group_id[module_name] = idx
        self._module2group_id: Dict[str, int] = module2group_id
        self._search_groups = search_groups

        self._search_space = None

    def prepare_from_supernet(self, supernet: BaseArchitecture) -> None:
        self._search_space = self._build_search_space(supernet)

    @property
    def search_space(self) -> Dict[int, List[DynamicMutable]]:
        if self._search_space is None:
            raise AttributeError(
                'Call `prepare_from_supernet` before access search space')
        return self._search_space

    def _build_search_space(
            self,
            supernet: BaseArchitecture) -> Dict[int, List[DynamicMutable]]:
        search_space = dict()
        group_nums = len(self._search_groups)
        for name, module in supernet.named_modules():
            if isinstance(module, DynamicMutable):
                group_id = self._module2group_id.get(name)
                # dynamic mutable that not in search groups
                if group_id is None:
                    group_id = group_nums
                    group_nums += 1
                try:
                    search_space[group_id].append(module)
                except KeyError:
                    search_space[group_id] = [module]

        return search_space

    @property
    def max_subnet(self) -> Dict[int, Any]:
        max_subnet = dict()
        for group_id, modules in self.search_space.items():
            max_subnet[group_id] = modules[0].max_choice

        return max_subnet

    @property
    def min_subnet(self) -> Dict[int, Any]:
        min_subnet = dict()
        for group_id, modules in self.search_space.items():
            min_subnet[group_id] = modules[0].min_choice
        return min_subnet

    @property
    def random_subnet(self) -> Dict[int, Any]:
        random_subnet = dict()
        for group_id, modules in self.search_space.items():
            random_subnet[group_id] = modules[0].random_choice

        # FIXME
        # dist.broadcast_object_list()
        return random_subnet

    def set_subnet(self, subnet_dict: Dict[int, Any]) -> None:
        for group_id, modules in self.search_space.items():
            choice = subnet_dict[group_id]
            for module in modules:
                module.set_choice(choice)
