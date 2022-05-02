_base_ = [
    '../../_base_/mmcls_runtime.py'
    '../../_base_/datasets/pipelines/rand_aug.py',
    '../../_base_/schedules/mmcls/imagenet_bs1024_aa_bignas.py',
]

_samples_per_gpu = 512
_number_of_gpus = 8
_batch_size = _samples_per_gpu * _number_of_gpus

_train_dataset_size = 1281167
_iterations_per_epoch = int(_train_dataset_size / _batch_size)

_initial_lr = (0.256 / 4096) * (_batch_size)
_min_lr = _initial_lr * 0.05

# dataset settings
dataset_type = 'ImageNet'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
file_client_args = dict(
    backend='petrel',
    path_mapping=dict({
        'data/imagenet/':
        'openmmlab:s3://openmmlab/datasets/classification/imagenet/',
        'data/imagenet/':
        'openmmlab:s3://openmmlab/datasets/classification/imagenet/'
    }))
train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='RandomResizedCrop', size=224, backend='pillow'),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(
        type='RandAugment',
        policies={{_base_.rand_policies}},
        num_policies=2,
        total_level=10,
        magnitude_level=9,
        magnitude_std=0.5,
        hparams=dict(
            pad_val=[round(x) for x in img_norm_cfg['mean'][::-1]],
            interpolation='bilinear')),
    dict(
        type='RandomErasing',
        erase_prob=0.2,
        max_area_ratio=1 / 3,
        mode='rand',
        fill_color=tuple(img_norm_cfg['mean'][::-1]),
        fill_std=tuple(img_norm_cfg['std'][::-1])),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='Resize', size=(256, -1), backend='pillow'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=_samples_per_gpu,
    workers_per_gpu=16,
    train=dict(
        type=dataset_type,
        data_prefix='data/imagenet/train',
        ann_file='data/imagenet/meta/train.txt',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_prefix='data/imagenet/val',
        ann_file='data/imagenet/meta/val.txt',
        pipeline=test_pipeline),
    test=dict(
        # replace `data/val` with `data/test` for standard test
        type=dataset_type,
        data_prefix='data/imagenet/val',
        ann_file='data/imagenet/meta/val.txt',
        pipeline=test_pipeline))

optimizer = dict(lr=_initial_lr)
lr_config = dict(
    step=int(_iterations_per_epoch * 2.4),
    warmup_iters=_iterations_per_epoch * 3,
    warmup_ratio=1e-6 / _initial_lr,
    min_lr=_min_lr)
runner = dict(type='IterBasedRunner', max_iters=450 * _iterations_per_epoch)
checkpoint_config = dict(
    interval=_iterations_per_epoch * 10, max_keep_ckpts=10)
evaluation = dict(
    interval=_iterations_per_epoch * 10, metric='accuracy', save_best='auto')
custom_hooks = [
    dict(
        type='mmrazor.LinearMomentumEMAHook',
        momentum=1e-4,
        priority='ABOVE_NORMAL')
]

se_cfg = dict(
    ratio=4,
    act_cfg=(dict(type='HSwish'),
             dict(
                 type='HSigmoid', bias=3, divisor=6, min_value=0,
                 max_value=1)))
norm_cfg = dict(type='BN')
model_cfg = dict(
    type='mmcls.ImageClassifier',
    backbone=dict(
        type='BigNASMobileNet',
        first_channels=40,
        last_channels=1408,
        widen_factor=1.0,
        norm_cfg=norm_cfg,
        se_cfg=se_cfg),
    neck=dict(type='GlobalAveragePooling'),
    # TODO
    # dropout before linear in spring
    head=dict(
        type='mmrazor.LinearClsHead',
        in_channels=1408,
        num_classes=1000,
        dropout_rate=0.2,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        init_cfg=dict(
            type='Normal', layer='Linear', mean=0., std=0.01, bias=0.),
        topk=(1, 5)))

_specials = [
    dict(in_key='expand_conv', refer='parent', expand_ratio=6),
    dict(in_key='depthwise_conv', refer='parent', expand_ratio=1),
    dict(in_key='se', refer='parent', expand_ratio=1)
]

pruner = dict(
    type='RangePruner',
    except_start_keys=['head._layers'],
    # TODO
    # must be ordered
    range_config=dict(
        conv1=dict(start_key='backbone.conv1', min_channels=32, priority=2),
        layer1=dict(start_key='backbone.layer1', min_channels=16),
        layer2=dict(
            start_key='backbone.layer2',
            min_channels=24,
            # HACK
            # must be ordered
            specials=_specials),
        layer3=dict(
            start_key='backbone.layer3', min_channels=40, specials=_specials),
        layer4=dict(
            start_key='backbone.layer4', min_channels=80, specials=_specials),
        layer5=dict(
            start_key='backbone.layer5', min_channels=112, specials=_specials),
        layer6=dict(
            start_key='backbone.layer6', min_channels=192, specials=_specials),
        layer7=dict(
            start_key='backbone.layer7', min_channels=320, specials=_specials),
        conv2=dict(start_key='backbone.conv2', min_channels=1280)))

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

algorithm = dict(
    type='BigNAS',
    resizer_config=dict(
        shape_list=[192, 224, 288, 320], interpolation_type='bicubic'),
    architecture=dict(type='MMClsArchitecture', model_cfg=model_cfg),
    mutator=dict(type='DynamicMutator', search_groups=search_groups),
    pruner=pruner,
    distiller=dict(
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
        ]),
    is_supernet_training=False)

use_ddp_wrapper = True
optimizer_config = None
