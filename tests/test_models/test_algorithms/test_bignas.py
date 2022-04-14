# Copyright (c) OpenMMLab. All rights reserved.
import random

import pytest
import torch

from mmrazor.models.algorithms.bignas import _InputResizer


def _test_input_resizer_init() -> None:
    shape_list = [3, (4, 4), 5, 6]
    target_shape_list = [(3, 3), (4, 4), (5, 5), (6, 6)]
    resizer = _InputResizer(shape_list=shape_list)
    assert resizer.current_shape == (6, 6)
    assert resizer.shape_list == target_shape_list
    assert resizer._shape_set == set(target_shape_list)

    shape_list = [3, (2, 4), 5, 6]
    target_shape_list = [(2, 4), (3, 3), (5, 5), (6, 6)]
    resizer = _InputResizer(shape_list=shape_list)
    assert resizer.shape_list == target_shape_list
    assert resizer._shape_set == set(target_shape_list)

    shape_list = [(1, 2, 3)]
    with pytest.raises(ValueError):
        resizer = _InputResizer(shape_list=shape_list)

    shape_list = [3, 4, 5]
    for _ in range(10):
        interpolation_type = random.choice(
            list(_InputResizer.valid_interpolation_type))
        resizer = _InputResizer(
            shape_list=shape_list, interpolation_type=interpolation_type)
        assert resizer.current_interpolation_type == interpolation_type

    interpolation_type = 'wrong-interpolation-shape'
    with pytest.raises(ValueError):
        resizer = _InputResizer(
            shape_list=shape_list, interpolation_type=interpolation_type)


def _test_input_resizer_api() -> None:
    shape_list = [192, 224, 288, 320]
    resizer = _InputResizer(shape_list=shape_list)
    for shape in shape_list:
        resizer.set_shape(shape)
    with pytest.raises(ValueError):
        resizer.set_shape(123)
    resizer.set_max_shape()
    assert resizer.current_shape == (320, 320)
    resizer.set_min_shape()
    assert resizer.current_shape == (192, 192)
    for _ in range(10):
        resizer.set_random_shape()
        resizer.current_shape in shape_list

    fake_input = torch.rand(3, 3, 224, 224)
    resizer.set_min_shape()
    assert resizer.resize(fake_input).shape == torch.Size((3, 3, 192, 192))
    resizer.set_max_shape()
    assert resizer.resize(fake_input).shape == torch.Size((3, 3, 320, 320))
    for _ in range(10):
        resizer.set_random_shape()
        assert resizer.resize(fake_input).shape == \
            torch.Size((3, 3, *resizer.current_shape))


def test_input_resizer() -> None:
    _test_input_resizer_init()
    _test_input_resizer_api()
