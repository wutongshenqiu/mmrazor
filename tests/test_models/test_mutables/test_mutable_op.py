# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmrazor.models import MUTABLES
from mmrazor.models.mutables import DynamicKernelConv2d


@pytest.mark.parametrize('dynamic_mode', ['progressive', 'center_crop', '123'])
def test_dynamic_kernel_conv2d(dynamic_mode: str) -> None:
    cfg = dict(
        type='DynamicKernelConv2d',
        in_channels=3,
        out_channels=32,
        kernel_size_list=[5, 3, 3, 7],
        dynamic_mode=dynamic_mode)

    if dynamic_mode not in DynamicKernelConv2d.valid_dynamic_mode:
        with pytest.raises(AssertionError):
            dynamic_conv2d: DynamicKernelConv2d = MUTABLES.build(cfg)
        return

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
    opt = torch.optim.SGD(dynamic_conv2d.parameters(), lr=1)
    dynamic_conv2d.set_choice(3)
    out = dynamic_conv2d(x)
    assert out.shape == (10, 32, 64, 64)
    y = out.mean()
    opt.zero_grad()
    y.backward()
    grad = dynamic_conv2d.weight.grad
    assert grad is not None
    not_zero_grad = grad[:, :, 2:5, 2:5]
    assert not_zero_grad.eq(0).sum() == 0
    grad[:, :, 2:5, 2:5] = 0.
    assert grad.sum() == 0

    dynamic_conv2d.set_choice(5)
    out = dynamic_conv2d(x)
    assert out.shape == (10, 32, 64, 64)
    y = out.mean()
    opt.zero_grad()
    y.backward()
    grad = dynamic_conv2d.weight.grad
    assert grad is not None
    not_zero_grad = grad[:, :, 1:6, 1:6]
    assert not_zero_grad.eq(0).sum() == 0
    grad[:, :, 1:6, 1:6] = 0.
    assert grad.sum() == 0

    dynamic_conv2d.set_choice(7)
    out = dynamic_conv2d(x)
    assert out.shape == (10, 32, 64, 64)
    y = out.mean()
    opt.zero_grad()
    y.backward()
    grad = dynamic_conv2d.weight.grad
    assert grad is not None
    not_zero_grad = grad[:, :, :, :]
    assert not_zero_grad.eq(0).sum() == 0
    grad[:, :, :, :] = 0.
    assert grad.sum() == 0

    subnet_config['all_choices'].append(1)
    with pytest.raises(AssertionError):
        dynamic_conv2d.deploy_subnet(subnet_config)
    subnet_config['all_choices'] = subnet_config['all_choices'][:-1]
    subnet_config['subnet_choice'] = 4
    with pytest.raises(AssertionError):
        dynamic_conv2d.deploy_subnet(subnet_config)

    subnet_config['subnet_choice'] = 3
    dynamic_conv2d.deploy_subnet(subnet_config)

    assert dynamic_conv2d.is_deployed
    if dynamic_mode == 'progressive':
        for transform_matrix_name in \
                dynamic_conv2d._transform_matrix_name_list:
            assert not hasattr(dynamic_conv2d, transform_matrix_name)
    assert dynamic_conv2d.padding == 1
    assert dynamic_conv2d.weight.shape == (32, 3, 3, 3)

    assert dynamic_conv2d(x).shape == (10, 32, 64, 64)
