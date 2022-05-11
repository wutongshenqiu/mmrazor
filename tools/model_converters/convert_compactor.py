# Copyright (c) OpenMMLab. All rights reserved.
import argparse
from pathlib import Path
from typing import Dict

import torch
from mmcv import Config, fileio
from mmcv.runner import load_checkpoint, save_checkpoint
from torch import nn
from torch.nn import functional as F

from mmrazor.models import build_algorithm
from mmrazor.models.pruners import ResRepPruner
from mmrazor.models.pruners.utils import replace_module


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert compactor used in ResRep')
    parser.add_argument('config', type=Path, help='path of train config file')
    parser.add_argument('checkpoint', type=str, help='checkpoint path')
    parser.add_argument('--output-dir', type=Path, default='work_dirs/resrep')
    args = parser.parse_args()

    return args


def fuse_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    conv_w = conv.weight
    conv_b = conv.bias if conv.bias is not None else torch.zeros_like(
        bn.running_mean)

    factor = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    conv.weight = nn.Parameter(conv_w *
                               factor.reshape([conv.out_channels, 1, 1, 1]))
    conv.bias = nn.Parameter((conv_b - bn.running_mean) * factor + bn.bias)

    return conv


def _fold_conv(conv: nn.Conv2d,
               filtered_compactor_weight: torch.Tensor) -> nn.Conv2d:
    conv_weight = conv.weight
    new_conv_weight = F.conv2d(
        conv_weight.permute(1, 0, 2, 3),
        filtered_compactor_weight).permute(1, 0, 2, 3)

    conv_bias = conv.bias
    new_conv_bias = torch.zeros(filtered_compactor_weight.shape[0])
    for i in range(new_conv_bias.shape[0]):
        new_conv_bias[i] = conv_bias.dot(filtered_compactor_weight[i, :, 0, 0])

    conv.weight = nn.Parameter(new_conv_weight)
    conv.bias = nn.Parameter(new_conv_bias)

    return conv


@torch.no_grad()
def convert_compactor(pruner: ResRepPruner, model: nn.Module) -> Dict:
    channel_cfg = dict()

    for compactor_name, conv_names in pruner._compactor2modules.items():
        for conv_name in conv_names:
            bn_name = pruner._conv_norm_links[conv_name]
            conv_module = pruner.name2module[conv_name]
            bn_module = pruner.name2module[bn_name]
            fused_conv = fuse_conv_bn(conv_module, bn_module)

            compactor = pruner._compactors[compactor_name]
            filter_ids_l_threshold = compactor.filter_ids_l_threshold(
                pruner._threshold)

            raw_channels = compactor.weight.shape[0]
            pruned_channels = len(filter_ids_l_threshold)
            print(f'conv: {conv_name}, pruned channels: {pruned_channels}, '
                  f'total channels: {raw_channels}')
            compactor.weight[filter_ids_l_threshold] = 0
            folded_conv = _fold_conv(fused_conv, compactor.weight)
            print(f'Reset conv: {conv_name}')
            replace_module(model, conv_name, folded_conv)
            # TODO
            # replace bn layer with identity
            print(f'Replace bn: {bn_name} with identity')
            replace_module(model, bn_name, nn.Identity())

            try:
                channel_cfg['identity_layer'].append(bn_name)
            except KeyError:
                channel_cfg['identity_layer'] = [bn_name]
            raw_channels
            channel_cfg[conv_name] = {
                'raw_channels': raw_channels,
                'remain_channels': raw_channels - pruned_channels
            }
    delattr(pruner, '_compactors')

    print('convert compactors done.')

    return channel_cfg


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    algorithm = build_algorithm(cfg.algorithm)
    load_checkpoint(algorithm, args.checkpoint, map_location='cpu')
    pruner = getattr(algorithm, 'pruner')
    assert pruner is not None and isinstance(pruner, ResRepPruner)

    channel_cfg = convert_compactor(pruner, algorithm.architecture.model)
    fileio.dump(channel_cfg, args.output_dir / 'channel_cfg.yaml')
    save_checkpoint(algorithm, str(args.output_dir / 'resrep_pruned.pth'))


if __name__ == '__main__':
    main()
