# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any

import torch
import torch.distributed as dist
from mmengine import autocast
from mmengine.runner import Runner
from torch import Tensor
from torch.nn.modules.batchnorm import _BatchNorm
from torch.utils.data import DataLoader


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.avg: Tensor = 0
        self.sum: Tensor = 0
        self.count: int = 0

    def update(self, val: Any, n: int = 1) -> None:
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self) -> None:
        dist.all_reduce(self.sum, dist.ReduceOp.SUM, async_op=False)

        count = torch.tensor([self.count], device=self.sum.device)
        dist.all_reduce(count, dist.ReduceOp.SUM, async_op=False)
        self.count = count.item()

        self.avg = self.sum / self.count


class CalibrateBNMixin:
    runner: Runner
    fp16: bool

    @torch.no_grad()
    def calibrate_bn_statistics(self,
                                dataloader: DataLoader,
                                calibrated_sample_nums: int = 2000) -> None:

        def record_input_statistics_hook(bn_module: _BatchNorm, input: Tensor,
                                         output: Tensor) -> None:
            mean_average_meter: AverageMeter = bn_module.__mean_average_meter__
            var_average_meter: AverageMeter = bn_module.__var_average_meter__

            real_input = input[0]
            mean = real_input.mean((0, 2, 3))
            var = real_input.var((0, 2, 3), unbiased=True)

            mean_average_meter.update(mean, real_input.size(0))
            var_average_meter.update(var, real_input.size(0))
            if dist.is_available():
                mean_average_meter.all_reduce()
                var_average_meter.all_reduce()

        hook_handles = []

        for name, module in self.runner.model.named_modules():
            if isinstance(module, _BatchNorm):
                self.runner.logger.debug(
                    'register `record_input_statistics_hook` to module: '
                    f'{name}')
                module.__mean_average_meter__ = AverageMeter()
                module.__var_average_meter__ = AverageMeter()
                handle = module.register_forward_hook(
                    record_input_statistics_hook)
                hook_handles.append(handle)

        self.runner.model.eval()

        self.runner.logger.debug('start calibrating batch norm statistics')
        self.runner.logger.debug(
            f'total sample numbers for calibration: {calibrated_sample_nums}')
        remaining = calibrated_sample_nums
        for data_batch in dataloader:
            if len(data_batch) >= remaining:
                data_batch = data_batch[:remaining]
            remaining -= len(data_batch)
            self.runner.logger.debug(
                f'remaining samples for calibration: {remaining}')
            with autocast(enabled=self.fp16):
                self.runner.model.test_step(data_batch)

            if remaining <= 0:
                break

        for name, module in self.runner.model.named_modules():
            if isinstance(module, _BatchNorm):
                mean_average_meter = module.__mean_average_meter__
                var_average_meter = module.__var_average_meter__
                calibrated_bn_mean = mean_average_meter.avg
                calibrated_bn_var = var_average_meter.avg

                feature_dim = calibrated_bn_mean.size(0)

                self.runner.logger.debug(
                    f'layer: {name}, '
                    f'current feature dimension: {feature_dim}, '
                    f'number of samples for calibration: {mean_average_meter.count}'
                    f'l2 norm of calibrated running mean: {calibrated_bn_mean.norm()}, '
                    f'l2 norm of calibrated running var: {calibrated_bn_var.norm()}, '
                    f'l2 norm of original running mean: {module.running_mean[:feature_dim].norm()}, '
                    f'l2 norm of original running var: {module.running_var[:feature_dim].norm()}, '
                )

                module.running_mean[:feature_dim].copy_(calibrated_bn_mean)
                module.running_var[:feature_dim].copy_(calibrated_bn_var)

                del module.__mean_average_meter__
                del module.__var_average_meter__

        self.runner.logger.debug('remove all hooks for calibration')
        for handle in hook_handles:
            handle.remove()
