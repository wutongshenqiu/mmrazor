# Copyright (c) OpenMMLab. All rights reserved.
import pytest

from mmrazor.models.builder import ARCHITECTURES, MUTATORS


def test_dynamic_mutator() -> None:
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

    architecture_cfg = dict(
        type='MMClsArchitecture',
        model_cfg=model_cfg,
    )

    architecture = ARCHITECTURES.build(architecture_cfg)

    mutator_cfg_no_search_group = dict(type='DynamicMutator')
    mutator = MUTATORS.build(mutator_cfg_no_search_group)
    with pytest.raises(AttributeError):
        _ = mutator.search_space

    mutator.prepare_from_supernet(architecture)
    for _, modules in mutator.search_space.items():
        assert len(modules) == 1

    max_subnet = mutator.max_subnet
    assert max_subnet.keys() == mutator.search_space.keys()
    for group_id, choice in max_subnet.items():
        assert choice == mutator.search_space[group_id][0].max_choice
    min_subnet = mutator.min_subnet
    assert max_subnet.keys() == mutator.search_space.keys()
    for group_id, choice in min_subnet.items():
        assert choice == mutator.search_space[group_id][0].min_choice
    random_subnet = mutator.random_subnet
    assert max_subnet.keys() == mutator.search_space.keys()
    for group_id, choice in random_subnet.items():
        assert choice in mutator.search_space[group_id][0].choices

    mutator.set_subnet(random_subnet)
    for group_id, choice in random_subnet.items():
        modules = mutator.search_space[group_id]
        for module in modules:
            assert choice == module.current_choice

    search_groups = [
        dict(modules=[
            'backbone.layer1.0.depthwise_conv.0',
            'backbone.layer1.1.depthwise_conv.0'
        ]),
        dict(modules=[
            'backbone.layer2.0.depthwise_conv.0',
            'backbone.layer2.1.depthwise_conv.0',
            'backbone.layer2.2.depthwise_conv.0'
        ]),
        dict(modules=[
            'backbone.layer3.0.depthwise_conv.0',
            'backbone.layer3.1.depthwise_conv.0',
            'backbone.layer3.2.depthwise_conv.0'
        ]),
        dict(modules=[
            'backbone.layer4.0.depthwise_conv.0',
            'backbone.layer4.1.depthwise_conv.0',
            'backbone.layer4.2.depthwise_conv.0',
            'backbone.layer4.3.depthwise_conv.0'
        ]),
        dict(modules=[
            'backbone.layer5.0.depthwise_conv.0',
            'backbone.layer5.1.depthwise_conv.0',
            'backbone.layer5.2.depthwise_conv.0',
            'backbone.layer5.3.depthwise_conv.0',
            'backbone.layer5.4.depthwise_conv.0',
            'backbone.layer5.5.depthwise_conv.0'
        ]),
        dict(modules=[
            'backbone.layer6.0.depthwise_conv.0',
            'backbone.layer6.1.depthwise_conv.0',
            'backbone.layer6.2.depthwise_conv.0',
            'backbone.layer6.3.depthwise_conv.0',
            'backbone.layer6.4.depthwise_conv.0',
            'backbone.layer6.5.depthwise_conv.0'
        ]),
        dict(modules=[
            'backbone.layer7.0.depthwise_conv.0',
            'backbone.layer7.1.depthwise_conv.0'
        ])
    ]
    group_id2nums = {0: 2, 1: 3, 2: 3, 3: 4, 4: 6, 5: 6, 6: 2}

    mutator_cfg_with_search_group = dict(
        type='DynamicMutator', search_groups=search_groups)
    mutator = MUTATORS.build(mutator_cfg_with_search_group)
    with pytest.raises(AttributeError):
        _ = mutator.search_space

    mutator.prepare_from_supernet(architecture)
    for group_id, modules in mutator.search_space.items():
        if group_id in group_id2nums:
            assert group_id2nums[group_id] == len(modules)
        else:
            assert len(modules) == 1

    max_subnet = mutator.max_subnet
    assert max_subnet.keys() == mutator.search_space.keys()
    for group_id, choice in max_subnet.items():
        assert choice == mutator.search_space[group_id][0].max_choice
    min_subnet = mutator.min_subnet
    assert max_subnet.keys() == mutator.search_space.keys()
    for group_id, choice in min_subnet.items():
        assert choice == mutator.search_space[group_id][0].min_choice
    random_subnet = mutator.random_subnet
    assert max_subnet.keys() == mutator.search_space.keys()
    for group_id, choice in random_subnet.items():
        assert choice in mutator.search_space[group_id][0].choices

    mutator.set_subnet(random_subnet)
    for group_id, choice in random_subnet.items():
        modules = mutator.search_space[group_id]
        for module in modules:
            assert choice == module.current_choice
