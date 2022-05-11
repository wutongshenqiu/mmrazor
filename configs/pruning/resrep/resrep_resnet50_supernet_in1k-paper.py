_base_ = ['../../_base_/mmcls_runtime.py']

_samples_per_gpu = 32
_number_of_gpus = 8
_batch_size = _samples_per_gpu * _number_of_gpus
_train_dataset_size = 1281167
_iterations_per_epoch = int(_train_dataset_size / _batch_size)

file_client_args = dict(
    backend='petrel',
    path_mapping=dict({
        'data/imagenet/':
        'sproject:s3://openmmlab/datasets/classification/imagenet/',
        'data/imagenet/':
        'sproject:s3://openmmlab/datasets/classification/imagenet/'
    }))

log_config = dict(interval=1001)

dataset_type = 'ImageNet'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    # dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', size=224, backend='pillow'),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    # dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(256, -1), backend='pillow'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=_samples_per_gpu,
    workers_per_gpu=8,
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
evaluation = dict(
    interval=1, metric_options={'topk': (1, 5)}, save_best='auto')
checkpoint_config = dict(interval=1, max_keep_ckpts=30)

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
        topk=(1, 5)),
    init_cfg=dict(
        type='Pretrained',
        checkpoint='checkpoint/resnet50-0676ba61-mmcls_style.pth'))

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
    before_update_mask_iter=_iterations_per_epoch * 5,
    mask_interval=200,
    retraining=True,
    channel_cfg='work_dirs/resrep/channel_cfg.yaml')

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
workflow = [('train', 1)]
use_ddp_wrapper = True
log_level = 'DEBUG'
