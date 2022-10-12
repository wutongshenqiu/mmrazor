_base_ = ['./attentive_mobilenet_supernet_32xb64_in1k.py']

train_cfg = dict(
    _delete_=True,
    type='mmrazor.EvolutionSearchLoop',
    dataloader=_base_.val_dataloader,
    evaluator=_base_.val_evaluator,
    max_epochs=20,
    num_candidates=50,
    top_k=10,
    num_mutation=25,
    num_crossover=25,
    mutate_prob=0.1,
    flops_range=None,
    score_key='accuracy/top1')

load_from = '/mnt/lustre/sunyue1/autolink/workspace-547/0802_mmrazor/.base/bignas_supernet.pth'