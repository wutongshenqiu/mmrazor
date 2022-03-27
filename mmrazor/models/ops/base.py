# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC, abstractmethod
from typing import Any, Generic, List, TypeVar

from mmcv.runner import BaseModule


class BaseOP(BaseModule):
    """Base class for searchable operations.

    Args:
        in_channels (int): The input channels of the operation.
        out_channels (int): The output channels of the operation.
        stride (int): Stride of the operation. Defaults to 1.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int = 1,
                 **kwargs: Any) -> None:
        super(BaseOP, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride


CHOICE_TYPE = TypeVar('CHOICE_TYPE')


class BaseDynamicOP(ABC, Generic[CHOICE_TYPE]):

    def __init__(self, choices: List[CHOICE_TYPE]) -> None:
        self._choices = sorted(list(set(choices)))
        print(f'{type(self).__name__} with choices: {self._choices}')

    @abstractmethod
    def set_choice(self, choice: CHOICE_TYPE) -> None:
        """Set certain choice for dynamic op.

        Args:
            choice (CHOICE_TYPE): _description_
        """

    def choices(self) -> List[CHOICE_TYPE]:
        return self._choices

    @property
    @abstractmethod
    def current_choice(self) -> CHOICE_TYPE:
        """Return current active choice.

        Returns:
            CHOICE_TYPE: _description_
        """
