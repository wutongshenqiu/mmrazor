# Copyright (c) OpenMMLab. All rights reserved.
import random
from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, List, TypeVar

CHOICE_TYPE = TypeVar('CHOICE_TYPE')


class BaseMutable(ABC, Generic[CHOICE_TYPE]):

    @property
    @abstractmethod
    def is_deployed(self) -> bool:
        """Whether the mutable is deployed.

        Returns:
            bool: _description_
        """

    @abstractmethod
    def deploy_subnet(self, *args, **kwargs) -> None:
        """Deploy mutable with subnet config. After deploy, `is_deployed`
        property will be set to True.

        Returns:
            _type_: _description_
        """

    @abstractmethod
    def export_subnet(self) -> Any:
        """Export current subnet config, can be directly used to deploy config.

        Returns:
            Any: _description_
        """

    @property
    def num_choices(self) -> int:
        """Number of choices.

        Returns:
            int: _description_
        """
        return len(self.choices)

    @property
    @abstractmethod
    def choices(self) -> List[CHOICE_TYPE]:
        """List all choices.

        Returns:
            List[CHOICE_TYPE]: _description_
        """


class OneShotMutable(BaseMutable[CHOICE_TYPE]):

    def forward(self, x: Any) -> Any:
        if self.is_deployed:
            return self.forward_deploy(x)
        else:
            return self.forward_sample(x, self.current_choice)

    @abstractmethod
    def forward_deploy(self, x: Any) -> Any:
        """Forward when mutable is deployed."""

    @abstractmethod
    def forward_sample(self, x: Any, choice: CHOICE_TYPE) -> Any:
        """Forward with given choice when mutable needs to be sampled."""

    @abstractmethod
    def set_choice(self, choice: CHOICE_TYPE) -> None:
        """Set current choice, will affect `current_choice` property."""

    @property
    @abstractmethod
    def current_choice(self) -> CHOICE_TYPE:
        """Return current choice."""

    @abstractmethod
    def export_subnet(self) -> Dict:
        """Export subnet with current choice."""

    @abstractmethod
    def deploy_subnet(self, subnet_config: Dict) -> None:
        """Deploy subnet with given choice."""


class DynamicMutable(OneShotMutable[CHOICE_TYPE]):

    @property
    def min_choice(self) -> CHOICE_TYPE:
        return self.choices[-1]

    @property
    def max_choice(self) -> CHOICE_TYPE:
        return self.choices[0]

    @property
    def random_choice(self) -> CHOICE_TYPE:
        choice_idx = random.randint(0, self.num_choices - 1)

        return self.choices[choice_idx]

    def export_subnet(self) -> Dict[str, Any]:
        return dict(
            subnet_choice=self.current_choice, all_choices=self.choices)

    def get_subnet_choice(self, subnet_config: Dict[str, Any]) -> CHOICE_TYPE:
        """Extract choice of given subnet config."""
        subnet_choice = subnet_config.get('subnet_choice')
        if subnet_choice is None:
            raise ValueError(
                '`subnet_config` should contain `subnet_choice` key')

        all_choices = subnet_config.get('all_choices')
        if all_choices is not None:
            assert all_choices == self.choices
            assert subnet_choice in all_choices

        return subnet_choice
