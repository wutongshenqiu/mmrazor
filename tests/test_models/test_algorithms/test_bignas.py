# Copyright (c) OpenMMLab. All rights reserved.
import random

import pytest
import torch

from mmrazor.models.algorithms.bignas import BigNAS, _InputResizer
from mmrazor.models.builder import build_algorithm
from mmrazor.models.distillers import SelfDistiller
from mmrazor.models.mutators import DynamicMutator
from mmrazor.models.pruners import RangePruner


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


def test_bignas() -> None:
    model_cfg = dict(
        type='mmcls.ImageClassifier',
        backbone=dict(
            type='BigNASMobileNet',
            first_channels=40,
            last_channels=1408,
            widen_factor=1.0,
            norm_cfg=dict(type='BN')),
        neck=dict(type='GlobalAveragePooling'),
        head=dict(
            type='mmrazor.LinearClsHead',
            in_channels=1408,
            num_classes=1000,
            dropout_rate=0.2,
            loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
            init_cfg=dict(
                type='Normal', layer='Linear', mean=0., std=0.01, bias=0.),
            topk=(1, 5)))
    architecture_cfg = dict(type='MMClsArchitecture', model_cfg=model_cfg)

    bignas_cfg = dict(type='BigNAS', architecture_cfg=architecture_cfg)
    bignas = build_algorithm(bignas_cfg)

    assert isinstance(bignas, BigNAS)
    with pytest.raises(AttributeError):
        _ = bignas.mutator
    with pytest.raises(AttributeError):
        _ = bignas.pruner
    with pytest.raises(AttributeError):
        _ = bignas.distiller

    pruner_cfg = dict(
        type='RangePruner',
        except_start_keys=['head._layers'],
        # TODO
        # must be ordered
        range_config=dict(
            conv1=dict(
                start_key='backbone.conv1', min_channels=32, priority=2),
            layer1=dict(start_key='backbone.layer1', min_channels=16),
            layer2=dict(start_key='backbone.layer2', min_channels=24),
            layer3=dict(start_key='backbone.layer3', min_channels=40),
            layer4=dict(start_key='backbone.layer4', min_channels=80),
            layer5=dict(start_key='backbone.layer5', min_channels=112),
            layer6=dict(start_key='backbone.layer6', min_channels=192),
            layer7=dict(start_key='backbone.layer7', min_channels=320),
            conv2=dict(start_key='backbone.conv2', min_channels=1280)))
    mutator_cfg = dict(type='DynamicMutator')
    distiller_cfg = dict(
        type='SelfDistiller',
        components=[
            dict(
                student_module='head._layers.1',
                teacher_module='head._layers.1',
                losses=[
                    dict(
                        type='KLDivergence',
                        name='loss_kd',
                        tau=1,
                        loss_weight=1,
                    )
                ]),
        ])

    bignas_cfg = dict(
        type='BigNAS',
        resizer_config=dict(
            shape_list=[192, 224, 288, 320], interpolation_type='bicubic'),
        architecture_cfg=architecture_cfg,
        mutator_cfg=mutator_cfg,
        pruner_cfg=pruner_cfg,
        distiller_cfg=distiller_cfg,
        is_supernet_training=True)

    bignas: BigNAS = build_algorithm(bignas_cfg)

    assert isinstance(bignas.mutator, DynamicMutator)
    assert isinstance(bignas.pruner, RangePruner)
    assert isinstance(bignas.distiller, SelfDistiller)

    imgs = torch.randn(16, 3, 224, 224)
    label = torch.randint(0, 1000, (16, ))
    data = {'img': imgs, 'gt_label': label}
    opt = torch.optim.SGD(bignas.parameters(), lr=1)

    bignas._is_supernet_training = True
    outputs = bignas.train_step(data, opt)
    assert outputs['loss'].item() > 0
    assert outputs['num_samples'] == 16

    for p in bignas.parameters():
        assert p.grad.norm() > 0

    bignas._set_max_subnet()
    bignas._is_supernet_training = False
    outputs = bignas.train_step(data, opt)
    assert outputs['loss'].item() > 0
    assert outputs['num_samples'] == 16

    for p in bignas.parameters():
        assert p.grad.norm() > 0

    bignas._train_dropout(True)
    for module in bignas.modules():
        if isinstance(module, torch.nn.Dropout):
            assert module.training

    bignas._train_dropout(False)
    for module in bignas.modules():
        if isinstance(module, torch.nn.Dropout):
            assert not module.training
