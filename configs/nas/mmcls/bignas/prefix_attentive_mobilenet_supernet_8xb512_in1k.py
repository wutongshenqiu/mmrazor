_base_ = [
    'mmrazor::_base_/settings/imagenet_bs2048_autoslim_pil.py',
    'mmcls::_base_/default_runtime.py',
]

default_scope = 'mmcls'

# !dataset config
# ==========================================================================
# data preprocessor
data_preprocessor = dict(
    type='ImgDataPreprocessor',
    # RGB format normalization parameters
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    # convert image from BGR to RGB
    bgr_to_rgb=True,
)

supernet = dict(
    type='mmrazor.SearchableImageClassifier',
    backbone=dict(
        type='mmrazor.AttentiveMobileNet',
        first_out_channels_range=[16, 24, 8],
        last_out_channels_range=[1792, 1984, 1984 - 1792],
        act_cfg=dict(type='Swish')),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='mmrazor.DynamicLinearClsHead',
        num_classes=1000,
        in_channels=1984,
        loss=dict(
            type='LabelSmoothLoss',
            num_classes=1000,
            label_smooth_val=0.1,
            mode='original',
            loss_weight=1.0),
        topk=(1, 5)),
    input_resizer_cfg=dict(
        input_resizer=dict(type='mmrazor.DynamicInputResizer'),
        mutable_shape=dict(
            type='mmrazor.OneShotMutableValue',
            value_list=[(192, 192), (224, 224), (256, 256), (288, 288)],
            default_value=(224, 224))))

# !autoslim algorithm config
num_samples = 2
model = dict(
    type='mmrazor.BigNAS',
    num_samples=num_samples,
    architecture=supernet,
    data_preprocessor=data_preprocessor,
    distiller=dict(
        type='mmrazor.ConfigurableDistiller',
        teacher_recorders=dict(
            fc=dict(type='mmrazor.ModuleOutputs', source='head.fc')),
        student_recorders=dict(
            fc=dict(type='mmrazor.ModuleOutputs', source='head.fc')),
        distill_losses=dict(
            loss_kl=dict(type='mmrazor.KLDivergence', tau=1, loss_weight=1)),
        loss_forward_mappings=dict(
            loss_kl=dict(
                preds_S=dict(recorder='fc', from_student=True),
                preds_T=dict(recorder='fc', from_student=False)))),
    mutators=dict(
        channel_mutator=dict(type='mmrazor.BigNASChannelMutator'),
        value_mutator=dict(type='mmrazor.DynamicValueMutator')))

model_wrapper_cfg = dict(
    type='mmrazor.BigNASDDP',
    broadcast_buffers=False,
    find_unused_parameters=True)

optim_wrapper = dict(accumulative_counts=num_samples + 2)

# learning policy
max_epochs = 100
param_scheduler = dict(end=max_epochs)

# train, val, test setting
train_cfg = dict(max_epochs=max_epochs)
val_cfg = dict(type='mmrazor.AutoSlimValLoop', calibrated_sample_nums=4096)
test_cfg = dict(type='mmrazor.AutoSlimTestLoop', calibrated_sample_nums=4096)
