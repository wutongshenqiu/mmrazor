# Copyright (c) OpenMMLab. All rights reserved.
import random
from copy import deepcopy

import pytest
import torch

from mmrazor.models.builder import ARCHITECTURES, PRUNERS
from mmrazor.models.pruners.resrep_pruning import CompactorLayer


def test_ratio_pruner():
    model_cfg = dict(
        type='mmcls.ImageClassifier',
        backbone=dict(
            type='mmcls.ResNet',
            depth=18,
            num_stages=4,
            out_indices=(3, ),
            style='pytorch'),
        neck=dict(type='mmcls.GlobalAveragePooling'),
        head=dict(
            type='mmcls.LinearClsHead',
            num_classes=1000,
            in_channels=512,
            loss=dict(type='mmcls.CrossEntropyLoss', loss_weight=1.0),
            topk=(1, 5),
        ))

    architecture_cfg = dict(
        type='MMClsArchitecture',
        model=model_cfg,
    )

    pruner_cfg = dict(
        type='RatioPruner',
        ratios=[1 / 8, 2 / 8, 3 / 8, 4 / 8, 5 / 8, 6 / 8, 7 / 8, 1.0])

    imgs = torch.randn(16, 3, 224, 224)
    label = torch.randint(0, 1000, (16, ))

    architecture = ARCHITECTURES.build(architecture_cfg)
    pruner = PRUNERS.build(pruner_cfg)

    pruner.prepare_from_supernet(architecture)
    assert hasattr(pruner, 'channel_spaces')

    # test set_min_channel
    pruner_cfg_ = deepcopy(pruner_cfg)
    pruner_cfg_['ratios'].insert(0, 0)
    pruner_ = PRUNERS.build(pruner_cfg_)
    architecture_ = ARCHITECTURES.build(architecture_cfg)
    pruner_.prepare_from_supernet(architecture_)
    with pytest.raises(AssertionError):
        # Output channels should be a positive integer not zero
        pruner_.set_min_channel()

    # test set_max_channel
    pruner.set_max_channel()
    for name, module in architecture.model.named_modules():
        if hasattr(module, 'in_mask'):
            assert module.in_mask.sum() == module.in_mask.numel()
        if hasattr(module, 'out_mask'):
            assert module.out_mask.sum() == module.out_mask.numel()

    # test channel bins
    pruner.set_min_channel()
    channel_bins_dict = pruner.get_max_channel_bins(max_channel_bins=4)
    pruner.set_channel_bins(channel_bins_dict, 4)
    for name, module in architecture.model.named_modules():
        if hasattr(module, 'in_mask'):
            assert module.in_mask.sum() == module.in_mask.numel()
        if hasattr(module, 'out_mask'):
            assert module.out_mask.sum() == module.out_mask.numel()

    # test making groups logic
    subnet_dict = pruner.sample_subnet()
    assert isinstance(subnet_dict, dict)
    pruner.set_subnet(subnet_dict)
    subnet_dict = pruner.export_subnet()
    assert isinstance(subnet_dict, dict)
    pruner.deploy_subnet(architecture, subnet_dict)
    losses = architecture(imgs, return_loss=True, gt_label=label)
    assert losses['loss'].item() > 0


def test_resrep_compactor_layer() -> None:
    feature_nums = 10
    name = 'test'

    compactor = CompactorLayer(feature_nums, name)
    assert compactor.name == name
    assert hasattr(compactor, 'mask')
    assert compactor.mask.requires_grad is False
    assert hasattr(compactor, '_layer')
    assert type(getattr(compactor, '_layer')).__name__ == 'Conv2d'

    assert compactor.deactivated_filter_nums == 0

    x = torch.rand(10, feature_nums, 4, 4)
    assert torch.equal(x, compactor(x))

    y = compactor(x)
    z = y.mean()
    z.backward()
    assert compactor._layer.weight.grad is not None

    metric_list = compactor.get_metric_list()
    for metric in metric_list:
        assert isinstance(metric, float) and metric >= 0


def test_resrep_pruner() -> None:
    model_cfg = dict(
        type='mmcls.ImageClassifier',
        backbone=dict(
            type='ResNet',
            depth=50,
            num_stages=4,
            out_indices=(3, ),
            style='pytorch'),
        neck=dict(type='GlobalAveragePooling'),
        head=dict(
            type='LinearClsHead',
            num_classes=1000,
            in_channels=2048,
            loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
            topk=(1, 5),
        ))

    architecture_cfg = dict(type='MMClsArchitecture', model=model_cfg)

    pruner_cfg = dict(
        type='ResRepPruner',
        flops_constraint=2000000000,
        begin_granularity=4,
        lasso_strength=1e-4,
        input_shape=(3, 224, 224))

    imgs = torch.randn(16, 3, 224, 224)
    label = torch.randint(0, 1000, (16, ))

    architecture = ARCHITECTURES.build(architecture_cfg)
    losses = architecture(imgs, return_loss=True, gt_label=label)
    assert losses['loss'].item() > 0

    _test_resrep_pruner_init(pruner_cfg)
    _test_resrep_pruner_prepare_from_supernet(pruner_cfg, architecture_cfg)
    _test_resrep_pruner_modify_conv_forward(pruner_cfg, architecture_cfg)
    _test_resrep_pruner_update_mask(pruner_cfg, architecture_cfg)


def _test_resrep_pruner_init(pruner_cfg) -> None:
    pruner = PRUNERS.build(pruner_cfg)

    assert hasattr(pruner, '_flops_constraint')
    assert hasattr(pruner, '_begin_granularity')
    assert hasattr(pruner, '_least_channel_nums')
    assert hasattr(pruner, '_lasso_strength')
    assert hasattr(pruner, '_input_shape')


def _test_resrep_pruner_prepare_from_supernet(pruner_cfg,
                                              architecture_cfg) -> None:
    pruner = PRUNERS.build(pruner_cfg)
    architecture = ARCHITECTURES.build(architecture_cfg)

    pruner.prepare_from_supernet(architecture)
    # test _init_flops
    for _, module in architecture.named_modules():
        assert getattr(module, '__flops__') >= 0

    assert hasattr(pruner, '_compactors')
    for compactor_name, compactor in pruner._compactors.items():
        assert compactor_name == compactor.name

    # test module2compactor
    assert hasattr(pruner, '_module2compactor')
    assert len(pruner._module2compactor) == 2
    for module_name, compactor_name in pruner._module2compactor[
            'out_mask'].items():
        module = pruner.name2module[module_name]
        assert type(module).__name__ == 'Conv2d' and module.groups == 1
        assert module.__compactor_name__ == compactor_name
    for module_name, compactor_name in pruner._module2compactor[
            'in_mask'].items():
        module = pruner.name2module[module_name]
        parents = pruner.node2parents[module_name]
        parent = parents[0]
        parent_module = pruner.name2module[parent]
        assert type(
            parent_module).__name__ == 'Conv2d' and parent_module.groups == 1
        assert parent_module.__compactor_name__ == compactor_name


def _test_resrep_pruner_modify_conv_forward(pruner_cfg,
                                            architecture_cfg) -> None:
    pruner = PRUNERS.build(pruner_cfg)
    architecture = ARCHITECTURES.build(architecture_cfg)
    pruner.prepare_from_supernet(architecture)

    module_input_list = []
    module_output_list = []

    def save_input_output_hook(module, input, output) -> None:
        module_input_list.append(input)
        module_output_list.append(output)

    compactor_name = random.choice(list(pruner._compactors.keys()))
    compactor: CompactorLayer = pruner._compactors[compactor_name]
    hook = compactor.register_forward_hook(save_input_output_hook)

    imgs = torch.rand(16, 3, 224, 224)
    labels = torch.randint(0, 1000, (16, ))

    architecture.eval()
    with torch.no_grad():
        module_nums = len(pruner._compactor2modules[compactor.name])

        architecture(imgs, gt_label=labels)
        assert len(module_input_list) == len(module_output_list) == module_nums
        hook.remove()
        for i in range(module_nums):
            torch.equal(module_output_list[i],
                        compactor(*module_input_list[i]))

        compactor._layer.weight.copy_(torch.rand_like(compactor._layer.weight))
        compactor.register_forward_hook(save_input_output_hook)
        architecture(imgs, gt_label=labels)

        assert len(module_input_list) == len(module_output_list) == \
            module_nums << 1
        for i in range(module_nums):
            j = module_nums + i
            torch.equal(module_output_list[j],
                        compactor(*module_input_list[j]))
            torch.equal(
                torch.cat(module_input_list[i]),
                torch.cat(module_input_list[j]))
            torch.equal(
                compactor(*module_input_list[i]), module_output_list[j])


def _test_resrep_pruner_update_mask(pruner_cfg, architecture_cfg) -> None:
    pruner = PRUNERS.build(pruner_cfg)
    architecture = ARCHITECTURES.build(architecture_cfg)

    pruner.prepare_from_supernet(architecture)

    compactor_name = random.choice(list(pruner._compactors.keys()))
    compactor = pruner._compactors[compactor_name]

    with torch.no_grad():
        compactor._layer.weight[0, :, :, :] += 9999.
        compactor._layer.weight[1, :, :, :] += 999.
        compactor._layer.weight[2, :, :, :] += 99.
        compactor._layer.weight[3, :, :, :] += 9.

        metric_list = compactor.get_metric_list()
        metric_dict = pruner._get_metric_dict()

        for i in range(4):
            assert abs(metric_dict[(compactor_name, i)] -
                       metric_list[i]) < 1e-8

    with torch.no_grad():
        for i in range(pruner._begin_granularity + 10):
            compactor._layer.weight[i, :, :, :] = 0.

        compactors_mask = pruner.sample_subnet(architecture)
        for cn, compactor_mask in compactors_mask.items():
            if cn == compactor_name:
                assert pruner._get_mask_deactivated_filter_nums(
                    compactor_mask) == pruner._begin_granularity
            else:
                pruner._get_mask_deactivated_filter_nums(compactor_mask) == 0

        pruner.set_subnet(compactors_mask)

        for cn, c in pruner._compactors.items():
            assert torch.equal(c.mask, compactors_mask[cn])

        corresponding_modules = []
        for mn, cn in pruner._module2compactor['out_mask'].items():
            module = pruner.name2module[mn]
            if cn == compactor_name:
                corresponding_modules.append(mn)
                assert pruner._get_mask_deactivated_filter_nums(
                    module.out_mask) == pruner._begin_granularity
            else:
                assert pruner._get_mask_deactivated_filter_nums(
                    module.out_mask) == 0
        assert sorted(corresponding_modules) == \
            sorted(pruner._compactor2modules[compactor_name])

        for mn, cn in pruner._module2compactor['in_mask'].items():
            module = pruner.name2module[mn]
            if cn == compactor_name:
                assert pruner._get_mask_deactivated_filter_nums(
                    module.in_mask) == pruner._begin_granularity
            else:
                assert pruner._get_mask_deactivated_filter_nums(
                    module.in_mask) == 0
