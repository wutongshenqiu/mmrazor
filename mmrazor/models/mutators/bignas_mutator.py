# Copyright (c) OpenMMLab. All rights reserved.
from functools import partial

import numpy as np

from mmrazor.models.builder import MUTATORS
from mmrazor.models.mutables.mixins import DynamicMixin
from .base import BaseMutator


@MUTATORS.register_module()
class DynamicMutator(BaseMutator):

    def __init__(self, groups, **kwargs):
        super().__init__(**kwargs)
        self.groups = groups

    def build_search_space(self, supernet):

        search_space = dict()
        for name, module in supernet.named_modules():
            if isinstance(module, DynamicMixin):
                group_id = self.get_group_id(name)
                if group_id not in search_space:
                    search_space[group_id] = [module]
                else:
                    search_space[group_id].append(module)
        return search_space

    def get_group_id(self, module_name):

        group_id = module_name
        for idx, group in enumerate(self.groups):
            if module_name in group.modules:
                group_id = idx
                break

        return group_id

    @property
    def max_model(self):
        max_model = dict()
        for group_id, modules in self.search_space.items():
            max_model[group_id] = modules[0].max_choice
        return max_model

    @property
    def min_model(self):
        min_model = dict()
        for group_id, modules in self.search_space.items():
            min_model[group_id] = modules[0].min_choice
        return min_model

    @property
    def random_model(self):
        random_model = dict()
        for group_id, modules in self.search_space.items():
            num_choices = modules[0].num_choices
            probs = modules[0].choice_probs
            # TODO support num chosen > 1
            chosen = np.random.choice(num_choices, 1, p=probs)[0]
            random_model[group_id] = modules[0].choices[chosen]

        # TODO dist.broadcast_object_list()
        return random_model

    def set_subnet(self, model_dict):
        for group_id, modules in self.search_space.items():
            chosen = model_dict[group_id]
            for module in modules:
                module.forward = partial(module.forward, sampled=chosen)
