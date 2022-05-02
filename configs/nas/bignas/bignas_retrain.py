_base_ = ['./bignas.py']

algorithm = dict(
    is_supernet_training=False,
    channel_cfg_path='configs/nas/bignas/bignas_S_channel_cfg.yaml',
    architecture=dict(
        mutable_cfg_path='configs/nas/bignas/bignas_S_mutable_cfg.yaml'),
)

_samples_per_gpu = 512
_number_of_gpus = 8
_batch_size = _samples_per_gpu * _number_of_gpus

_train_dataset_size = 1281167
_iterations_per_epoch = int(_train_dataset_size / _batch_size)

runner = dict(type='IterBasedRunner', max_iters=600 * _iterations_per_epoch)
