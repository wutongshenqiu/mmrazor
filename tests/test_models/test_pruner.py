# Copyright (c) OpenMMLab. All rights reserved.
import random
from copy import deepcopy

import pytest
import torch
from mmcv import ConfigDict, digit_version

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

    _test_reset_bn_running_stats(architecture_cfg, pruner_cfg, False)
    with pytest.raises(AssertionError):
        _test_reset_bn_running_stats(architecture_cfg, pruner_cfg, True)

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

    # test models with shared module
    model_cfg = ConfigDict(
        type='mmdet.RetinaNet',
        backbone=dict(
            type='ResNet',
            depth=50,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            frozen_stages=1,
            norm_cfg=dict(type='BN', requires_grad=True),
            norm_eval=True,
            style='pytorch'),
        neck=dict(
            type='FPN',
            in_channels=[256, 512, 1024, 2048],
            out_channels=256,
            start_level=1,
            add_extra_convs='on_input',
            num_outs=5),
        bbox_head=dict(
            type='RetinaHead',
            num_classes=80,
            in_channels=256,
            stacked_convs=4,
            feat_channels=256,
            anchor_generator=dict(
                type='AnchorGenerator',
                octave_base_scale=4,
                scales_per_octave=3,
                ratios=[0.5, 1.0, 2.0],
                strides=[8, 16, 32, 64, 128]),
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[.0, .0, .0, .0],
                target_stds=[1.0, 1.0, 1.0, 1.0]),
            loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
        # model training and testing settings
        train_cfg=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.4,
                min_pos_iou=0,
                ignore_iof_thr=-1),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        test_cfg=dict(
            nms_pre=1000,
            min_bbox_size=0,
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100))

    architecture_cfg = dict(
        type='MMDetArchitecture',
        model=model_cfg,
    )

    pruner_cfg = dict(
        type='RatioPruner',
        ratios=[1 / 8, 2 / 8, 3 / 8, 4 / 8, 5 / 8, 6 / 8, 7 / 8, 1.0])

    architecture = ARCHITECTURES.build(architecture_cfg)
    pruner = PRUNERS.build(pruner_cfg)
    pruner.prepare_from_supernet(architecture)
    subnet_dict = pruner.sample_subnet()
    pruner.set_subnet(subnet_dict)
    subnet_dict = pruner.export_subnet()
    pruner.deploy_subnet(architecture, subnet_dict)
    architecture.forward_dummy(imgs)

    # test models with concat operations
    model_cfg = ConfigDict(
        type='mmdet.YOLOX',
        input_size=(640, 640),
        random_size_range=(15, 25),
        random_size_interval=10,
        backbone=dict(type='CSPDarknet', deepen_factor=0.33, widen_factor=0.5),
        neck=dict(
            type='YOLOXPAFPN',
            in_channels=[128, 256, 512],
            out_channels=128,
            num_csp_blocks=1),
        bbox_head=dict(
            type='YOLOXHead',
            num_classes=80,
            in_channels=128,
            feat_channels=128),
        train_cfg=dict(
            assigner=dict(type='SimOTAAssigner', center_radius=2.5)),
        # In order to align the source code, the threshold of the val phase is
        # 0.01, and the threshold of the test phase is 0.001.
        test_cfg=dict(
            score_thr=0.01, nms=dict(type='nms', iou_threshold=0.65)))

    architecture_cfg = dict(
        type='MMDetArchitecture',
        model=model_cfg,
    )

    architecture = ARCHITECTURES.build(architecture_cfg)
    pruner.prepare_from_supernet(architecture)
    subnet_dict = pruner.sample_subnet()
    pruner.set_subnet(subnet_dict)
    subnet_dict = pruner.export_subnet()
    pruner.deploy_subnet(architecture, subnet_dict)
    architecture.forward_dummy(imgs)

    # test models with groupnorm
    model_cfg = ConfigDict(
        type='mmdet.ATSS',
        backbone=dict(
            type='ResNet',
            depth=50,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            frozen_stages=1,
            norm_cfg=dict(type='BN', requires_grad=True),
            norm_eval=True,
            style='pytorch',
            init_cfg=dict(
                type='Pretrained', checkpoint='torchvision://resnet50')),
        neck=dict(
            type='FPN',
            in_channels=[256, 512, 1024, 2048],
            out_channels=256,
            start_level=1,
            add_extra_convs='on_output',
            num_outs=5),
        bbox_head=dict(
            type='ATSSHead',
            num_classes=80,
            in_channels=256,
            stacked_convs=4,
            feat_channels=256,
            anchor_generator=dict(
                type='AnchorGenerator',
                ratios=[1.0],
                octave_base_scale=8,
                scales_per_octave=1,
                strides=[8, 16, 32, 64, 128]),
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[.0, .0, .0, .0],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0),
            loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
            loss_centerness=dict(
                type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
        # training and testing settings
        train_cfg=dict(
            assigner=dict(type='ATSSAssigner', topk=9),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        test_cfg=dict(
            nms_pre=1000,
            min_bbox_size=0,
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.6),
            max_per_img=100))

    architecture_cfg = dict(
        type='MMDetArchitecture',
        model=model_cfg,
    )

    architecture = ARCHITECTURES.build(architecture_cfg)
    # ``StructurePruner`` requires pytorch>=1.6.0 to auto-trace GroupNorm
    # correctly
    min_required_version = '1.6.0'
    if digit_version(torch.__version__) < digit_version(min_required_version):
        with pytest.raises(AssertionError):
            pruner.prepare_from_supernet(architecture)
    else:
        pruner.prepare_from_supernet(architecture)
        subnet_dict = pruner.sample_subnet()
        pruner.set_subnet(subnet_dict)
        subnet_dict = pruner.export_subnet()
        pruner.deploy_subnet(architecture, subnet_dict)
        architecture.forward_dummy(imgs)


def _test_reset_bn_running_stats(architecture_cfg, pruner_cfg, should_fail):
    import os
    import random

    import numpy as np

    def set_seed(seed: int) -> None:
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    output_list = []

    def output_hook(self, input, output) -> None:
        output_list.append(output)

    set_seed(1024)

    imgs = torch.randn(16, 3, 224, 224)

    torch_rng_state = torch.get_rng_state()
    np_rng_state = np.random.get_state()
    random_rng_state = random.getstate()

    architecture1 = ARCHITECTURES.build(architecture_cfg)
    pruner1 = PRUNERS.build(pruner_cfg)
    if should_fail:
        pruner1._reset_norm_running_stats = lambda *_: None
    set_seed(1)
    pruner1.prepare_from_supernet(architecture1)
    architecture1.model.head.fc.register_forward_hook(output_hook)
    architecture1.eval()
    architecture1(imgs, return_loss=False)

    set_seed(1024)
    torch.set_rng_state(torch_rng_state)
    np.random.set_state(np_rng_state)
    random.setstate(random_rng_state)

    architecture2 = ARCHITECTURES.build(architecture_cfg)
    pruner2 = PRUNERS.build(pruner_cfg)
    if should_fail:
        pruner2._reset_norm_running_stats = lambda *_: None
    set_seed(2)
    pruner2.prepare_from_supernet(architecture2)
    architecture2.model.head.fc.register_forward_hook(output_hook)
    architecture2.eval()
    architecture2(imgs, return_loss=False)

    assert torch.equal(output_list[0].norm(p='fro'),
                       output_list[1].norm(p='fro'))


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
    _test_resrep_pruner_compactor_same_weights_after_init_weights(
        pruner_cfg, architecture_cfg)
    _test_resrep_pruner_when_insert_compactor(
        pruner_cfg, architecture_cfg,
        'checkpoint/resnet50_8xb32_in1k_20210831-ea4938fc.pth')
    _test_resrep_pruner_prepare_from_supernet(pruner_cfg, architecture_cfg)
    _test_resrep_pruner_modify_conv_forward(pruner_cfg, architecture_cfg)
    _test_resrep_pruner_update_mask(pruner_cfg, architecture_cfg)


def _test_resrep_pruner_init(pruner_cfg) -> None:
    pruner = PRUNERS.build(pruner_cfg)

    assert hasattr(pruner, '_flops_constraint')
    assert hasattr(pruner, '_flops_ratio')
    assert hasattr(pruner, '_begin_granularity')
    assert hasattr(pruner, '_least_channel_nums')
    assert hasattr(pruner, '_lasso_strength')
    assert hasattr(pruner, '_input_shape')


def _test_resrep_pruner_compactor_same_weights_after_init_weights(
        pruner_cfg, architecture_cfg) -> None:
    pruner = PRUNERS.build(pruner_cfg)
    architecture = ARCHITECTURES.build(architecture_cfg)
    pruner.prepare_from_supernet(architecture)

    pre_compactors_params = dict()
    for name, param in pruner._compactors.named_parameters():
        pre_compactors_params[name] = param.clone()

    pruner.init_weights()

    for name, param in pruner._compactors.named_parameters():
        assert torch.equal(pre_compactors_params[name], param)


def _test_resrep_pruner_when_insert_compactor(pruner_cfg,
                                              architecture_cfg,
                                              checkpoint_path=None) -> None:
    pruner = PRUNERS.build(pruner_cfg)
    architecture = ARCHITECTURES.build(architecture_cfg)

    imgs = torch.rand(16, 3, 224, 224)
    pre_y = architecture.forward_dummy(imgs)
    pruner.prepare_from_supernet(architecture)
    after_y = architecture.forward_dummy(imgs)
    assert torch.equal(pre_y, after_y)

    import os
    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        return
    architecture_cfg_ckpt = deepcopy(architecture_cfg)
    architecture_cfg_ckpt['model']['init_cfg'] = {
        'type': 'Pretrained',
        'checkpoint': checkpoint_path
    }
    architecture = ARCHITECTURES.build(architecture_cfg_ckpt)
    architecture.eval()
    model = architecture.model
    model.init_weights()
    pruner._reset_norm_running_stats(architecture)
    pre_y = architecture.forward_dummy(imgs)
    pruner.prepare_from_supernet(architecture)
    middle_y = architecture.forward_dummy(imgs)
    model.init_weights()
    pruner._reset_norm_running_stats(architecture)
    after_y = architecture.forward_dummy(imgs)
    assert torch.equal(pre_y, middle_y)
    assert torch.equal(middle_y, after_y)


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

    # test get metric list
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

    # test sample_subnet and set_subnet
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

    # test least_channel_nums
    pruner_cfg_lcn = deepcopy(pruner_cfg)
    least_channel_nums = random.randint(1, 10)
    pruner_cfg_lcn['least_channel_nums'] = least_channel_nums
    pruner_cfg_lcn['begin_granularity'] = 10000
    pruner = PRUNERS.build(pruner_cfg_lcn)
    architecture = ARCHITECTURES.build(architecture_cfg)
    pruner.prepare_from_supernet(architecture)
    pruner.update_mask(architecture)

    for compactor in pruner._compactors.values():
        assert pruner._get_mask_activated_filter_nums(compactor.mask) >= \
            least_channel_nums
