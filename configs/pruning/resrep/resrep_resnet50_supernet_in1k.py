_base_ = [
    '../../_base_/datasets/mmcls/imagenet_bs32.py',
    '../../_base_/mmcls_runtime.py'
]

# model settings
model = dict(
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

algorithm = dict(
    type='ResRep',
    architecture=dict(type='MMClsArchitecture', model=model),
    distiller=None,
    pruner=dict(
        type='ResRepPruner',
        flops_ratio=0.455,
        begin_granularity=4,
        ignore_skip_mask=True,
        lasso_strength=1e-4,
        input_shape=(3, 224, 224),
        follow_paper=True),
    before_update_mask_iter=25022,
    mask_interval=200)

# TODO
# same as pytorch?
lr_config = dict(policy='CosineAnnealing', min_lr=0, by_epoch=False)

runner = dict(type='EpochBasedRunner', max_epochs=180)

# TODO
# compactor has different momentum
optimizer = dict(
    architecture=dict(
        type='SGD',
        lr=0.01,
        momentum=0.9,
        weight_decay=1e-4,
        paramwise_cfg=dict(
            bias_lr_mult=2., bias_decay_mult=0., norm_decay_mult=0.)),
    pruner=dict(
        type='SGD',
        lr=0.01,
        momentum=0.99,
        weight_decay=0.,
        paramwise_cfg=dict(
            bias_lr_mult=2., bias_decay_mult=0., norm_decay_mult=0.)),
)
optimizer_config = None

evaluation = dict(
    interval=2, metric='accuracy', metric_options={'topk': (1, 5)}, start=1)

workflow = [('train', 1)]

checkpoint_config = dict(interval=1, max_keep_ckpts=5)

use_ddp_wrapper = True

log_level = 'DEBUG'

data = dict(samples_per_gpu=32)
