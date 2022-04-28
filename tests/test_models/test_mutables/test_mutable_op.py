# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmrazor.models import MUTABLES
from mmrazor.models.mutables import DynamicKernelConv2d


def test_dynamic_kernel_conv2d() -> None:
    cfg = dict(
        type='DynamicKernelConv2d',
        in_channels=3,
        out_channels=32,
        kernel_size_list=[5, 3, 3, 7])

    dynamic_conv2d: DynamicKernelConv2d = MUTABLES.build(cfg)
    assert dynamic_conv2d.in_channels == cfg['in_channels']
    assert dynamic_conv2d.out_channels == cfg['out_channels']
    assert dynamic_conv2d.choices == [7, 5, 3]
    assert not dynamic_conv2d.is_deployed
    assert dynamic_conv2d.current_choice == 7

    subnet_config = {'subnet_choice': 7, 'all_choices': [7, 5, 3]}
    assert dynamic_conv2d.export_subnet() == subnet_config

    with pytest.raises(AssertionError):
        dynamic_conv2d.set_choice(4)
    dynamic_conv2d.set_choice(5)
    assert dynamic_conv2d.current_choice == 5
    subnet_config['subnet_choice'] = 5
    assert dynamic_conv2d.export_subnet() == subnet_config

    assert dynamic_conv2d.max_choice == 7
    assert dynamic_conv2d.min_choice == 3
    for _ in range(10):
        assert dynamic_conv2d.random_choice in dynamic_conv2d.choices

    x = torch.rand(10, 3, 64, 64)
    dynamic_conv2d.set_choice(3)
    assert dynamic_conv2d(x).shape == (10, 32, 64, 64)
    dynamic_conv2d.set_choice(5)
    assert dynamic_conv2d(x).shape == (10, 32, 64, 64)
    dynamic_conv2d.set_choice(7)
    assert dynamic_conv2d(x).shape == (10, 32, 64, 64)

    subnet_config['all_choices'].append(1)
    with pytest.raises(AssertionError):
        dynamic_conv2d.deploy_subnet(subnet_config)
    subnet_config['all_choices'] = subnet_config['all_choices'][:-1]
    subnet_config['subnet_choice'] = 4
    with pytest.raises(AssertionError):
        dynamic_conv2d.deploy_subnet(subnet_config)

    subnet_config['subnet_choice'] = 3
    dynamic_conv2d.deploy_subnet(subnet_config)

    assert not dynamic_conv2d.is_deployed
    assert dynamic_conv2d.padding == 1
    assert dynamic_conv2d.weight.shape == (32, 3, 3, 3)

    assert dynamic_conv2d(x).shape == (10, 32, 64, 64)
