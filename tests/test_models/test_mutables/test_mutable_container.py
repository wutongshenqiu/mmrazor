# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmrazor.models import MUTABLES
from mmrazor.models.mutables import DynamicSequential


def test_dynamic_kernel_conv2d() -> None:
    dynamic_cfg1 = dict(
        type='DynamicKernelConv2d',
        in_channels=3,
        out_channels=32,
        kernel_size_list=[5, 3, 3, 7])
    dynamic_cfg2 = dict(
        type='DynamicKernelConv2d',
        in_channels=32,
        out_channels=32,
        kernel_size_list=[5, 3, 3, 7])
    ops = [
        MUTABLES.build(dynamic_cfg1),
        MUTABLES.build(dynamic_cfg2),
        MUTABLES.build(dynamic_cfg2),
        MUTABLES.build(dynamic_cfg2),
        MUTABLES.build(dynamic_cfg2),
        MUTABLES.build(dynamic_cfg2),
    ]

    with pytest.raises(AssertionError):
        dynamic_sequential = DynamicSequential(*ops, length_list=[5, 3, 3, 4])

    ops = ops[:-1]
    dynamic_sequential: DynamicSequential = DynamicSequential(
        *ops, length_list=[5, 3, 3, 4])

    assert dynamic_sequential.choices == [5, 4, 3]
    assert not dynamic_sequential.is_deployed

    subnet_config = {'subnet_choice': 5, 'all_choices': [5, 4, 3]}
    assert dynamic_sequential.export_subnet() == subnet_config

    with pytest.raises(AssertionError):
        dynamic_sequential.set_choice(2)
    dynamic_sequential.set_choice(4)
    assert dynamic_sequential.current_choice == 4
    subnet_config['subnet_choice'] = 4
    assert dynamic_sequential.export_subnet() == subnet_config

    assert dynamic_sequential.max_choice == 5
    assert dynamic_sequential.min_choice == 3
    for _ in range(10):
        assert dynamic_sequential.random_choice in dynamic_sequential.choices

    x = torch.rand(10, 3, 64, 64)
    dynamic_sequential.set_choice(3)
    assert dynamic_sequential(x).shape == (10, 32, 64, 64)
    dynamic_sequential.set_choice(4)
    assert dynamic_sequential(x).shape == (10, 32, 64, 64)
    dynamic_sequential.set_choice(5)
    assert dynamic_sequential(x).shape == (10, 32, 64, 64)

    subnet_config['all_choices'].append(1)
    with pytest.raises(AssertionError):
        dynamic_sequential.deploy_subnet(subnet_config)
    subnet_config['all_choices'] = subnet_config['all_choices'][:-1]
    subnet_config['subnet_choice'] = 6
    with pytest.raises(AssertionError):
        dynamic_sequential.deploy_subnet(subnet_config)

    subnet_config['subnet_choice'] = 3
    dynamic_sequential.deploy_subnet(subnet_config)
    assert dynamic_sequential.is_deployed
    assert len(dynamic_sequential) == 3

    assert dynamic_sequential(x).shape == (10, 32, 64, 64)
