_base_ = [
    '../../_base_/datasets/mmcls/imagenet_bs32.py',
    '../../_base_/schedules/mmcls/imagenet_bs256.py',
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
        flops_ratio=0.6,
        begin_granularity=4,
        lasso_strength=1e-4,
        input_shape=(3, 224, 224)),
    before_update_mask_iter=10000,
    mask_interval=200)

lr_config = dict(_delete_=True, policy='CosineAnnealing', min_lr=0)

runner = dict(type='EpochBasedRunner', max_epochs=180)

optimizer = dict(type='SGD', momentum=0.9, weight_decay=1e-4)
optimizer_config = None

evaluation = dict(
    interval=1, metric='accuracy', metric_options={'topk': (1, 5)})

use_ddp_wrapper = True
