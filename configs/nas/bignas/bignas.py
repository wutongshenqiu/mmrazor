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
model = dict(
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

mutator = dict(
    type='BigNASMutator',
    placeholder_mapping=dict(
        dynamic_conv2d_k3=dict(
            type='DynamicOP',
            choices=[3],
            dynamic_cfg=dict(type='DynamicConv2d')),
        dynamic_conv2d_k35=dict(
            type='DynamicOP',
            choices=[3, 5],
            dynamic_cfg=dict(type='DynamicConv2d')),
        stage0=dict(
            type='MutableSequential',
            length_list=list(range(1, 3)),
        ),
        stage1=dict(
            type='MutableSequential',
            length_list=list(range(2, 4)),
        ),
        stage2=dict(
            type='MutableSequential',
            length_list=list(range(2, 4)),
        ),
        stage3=dict(
            type='MutableSequential',
            length_list=list(range(2, 5)),
        ),
        stage4=dict(
            type='MutableSequential',
            length_list=list(range(2, 7)),
        ),
        stage5=dict(
            type='MutableSequential',
            length_list=list(range(2, 7)),
        ),
        stage6=dict(
            type='MutableSequential',
            length_list=list(range(1, 3)),
        ),
    ))

_specials = [
    dict(in_key='expand_conv', refer='parent', expand_ratio=6),
    dict(in_key='depthwise_conv', refer='parent', expand_ratio=1),
    dict(in_key='se', refer='parent', expand_ratio=1)
]

algorithm = dict(
    type='BigNAS',
    resizer_config=dict(
        shape_list=[192, 224, 288, 320], interpolation_type='bicubic'),
    architecture=dict(
        type='MMClsArchitecture',
        model=model,
    ),
    mutator=mutator,
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
    retraining=False,
    pruner=dict(
        type='RangePruner',
        except_start_keys=['head.layers'],
        # TODO
        # must be ordered
        range_config=dict(
            conv1=dict(
                start_key='backbone.conv1', min_channels=32, priority=2),
            layer1=dict(start_key='backbone.layer1', min_channels=16),
            layer2=dict(
                start_key='backbone.layer2',
                min_channels=24,
                # HACK
                # must be ordered
                specials=_specials),
            layer3=dict(
                start_key='backbone.layer3',
                min_channels=40,
                specials=_specials),
            layer4=dict(
                start_key='backbone.layer4',
                min_channels=80,
                specials=_specials),
            layer5=dict(
                start_key='backbone.layer5',
                min_channels=112,
                specials=_specials),
            layer6=dict(
                start_key='backbone.layer6',
                min_channels=192,
                specials=_specials),
            layer7=dict(
                start_key='backbone.layer7',
                min_channels=320,
                specials=_specials),
            conv2=dict(start_key='backbone.conv2', min_channels=1280))))

use_ddp_wrapper = True
