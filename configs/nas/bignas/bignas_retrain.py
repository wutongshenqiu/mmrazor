_base_ = ['./bignas.py']

algorithm = dict(
    architecture=dict(
        mutable_cfg='configs/nas/bignas/bignas_M_mutator_cfg.yaml'),
    distiller=None,
    retraining=True)

_samples_per_gpu = 96
_number_of_gpus = 8
_batch_size = _samples_per_gpu * _number_of_gpus

_train_dataset_size = 1281167
_iterations_per_epoch = int(_train_dataset_size / _batch_size)

runner = dict(type='IterBasedRunner', max_iters=600 * _iterations_per_epoch)
