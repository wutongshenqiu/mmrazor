_base_ = [
    '../../_base_/datasets/mmcls/imagenet_bs128_colorjittor_aa_bignas.py',
    '../../_base_/schedules/mmcls/imagenet_bs1024_aa_bignas.py',
    '../../_base_/mmcls_runtime.py'
]

_samples_per_gpu = 96
_number_of_gpus = 8
_batch_size = _samples_per_gpu * _number_of_gpus

_train_dataset_size = 1281167
_iterations_per_epoch = (_train_dataset_size / _batch_size).__ceil__()

_initial_lr = (0.256 / 4096) * (_batch_size)
_min_lr = _initial_lr * 0.05

data = dict(samples_per_gpu=_samples_per_gpu)
optimizer = dict(lr=_initial_lr)
lr_config = dict(
    step=int(_iterations_per_epoch * 2.4),
    warmup_iters=_iterations_per_epoch * 3,
    warmup_ratio=1e-6 / _initial_lr,
    min_lr=_min_lr)
runner = dict(type='IterBasedRunner', max_iters=450 * _iterations_per_epoch)
checkpoint_config = dict(interval=20000, max_keep_ckpts=5)
evaluation = dict(interval=5000, metric='accuracy')

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
    retraining=False)

use_ddp_wrapper = True
