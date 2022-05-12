_base_ = ['resrep_resnet50_supernet_in1k-paper.py']

algorithm = dict(
    channel_cfg='work_dirs/resrep/channel_cfg.yaml', retraining=True)
