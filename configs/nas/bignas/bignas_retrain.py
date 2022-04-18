_base_ = ['./bignas.py']

algorithm = dict(
    retraining=True,
    channel_cfg='configs/nas/bignas/bignas_M_channel_cfg.yaml',
    mutable_cfg='configs/nas/bignas/bignas_M_mutator_cfg.yaml')

_samples_per_gpu = 96
_number_of_gpus = 8
_batch_size = _samples_per_gpu * _number_of_gpus

_train_dataset_size = 1281167
_iterations_per_epoch = (_train_dataset_size / _batch_size).__ceil__()

runner = dict(type='IterBasedRunner', max_iters=600 * _iterations_per_epoch)
