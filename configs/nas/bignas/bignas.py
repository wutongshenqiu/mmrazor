_base_ = [
    '../../_base_/datasets/mmcls/imagenet_bs128_colorjittor.py',
    '../../_base_/schedules/mmcls/imagenet_bs1024_spos.py',
    '../../_base_/mmcls_runtime.py'
]
norm_cfg = dict(type='BN')
model = dict(
    type='mmcls.ImageClassifier',
    backbone=dict(
        type='BigNASMobileNet',
        first_channels=40,
        last_channels=1408,
        widen_factor=1.0,
        norm_cfg=norm_cfg),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=1408,
        loss=dict(
            type='LabelSmoothLoss',
            num_classes=1000,
            label_smooth_val=0.1,
            mode='original',
            loss_weight=1.0),
        topk=(1, 5),
    ),
)

mutator = dict(
    type='BigNASMutator',
    placeholder_mapping=dict(
        dynamic_conv2d_k3=dict(
            type='DynamicOP',
            choices=[3],
            dynamic_cfg=dict(type='DynamicConv2d', )),
        dynamic_conv2d_k35=dict(
            type='DynamicOP',
            choices=[3, 5],
            dynamic_cfg=dict(type='DynamicConv2d', )),
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

algorithm = dict(
    type='BigNAS',
    architecture=dict(
        type='MMClsArchitecture',
        model=model,
    ),
    mutator=mutator,
    distiller=dict(
        type='SelfDistiller',
        components=[
            dict(
                student_module='head.fc',
                teacher_module='head.fc',
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
        type='RatioPruner',
        ratios=(2 / 12, 3 / 12, 4 / 12, 5 / 12, 6 / 12, 7 / 12, 8 / 12, 9 / 12,
                10 / 12, 11 / 12, 1.0)))

runner = dict(type='EpochBasedRunner', max_epochs=50)
evaluation = dict(interval=1, metric='accuracy')

# checkpoint saving
checkpoint_config = dict(interval=5)

use_ddp_wrapper = True
