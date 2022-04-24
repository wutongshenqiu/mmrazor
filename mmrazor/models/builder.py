# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy

from mmcv.cnn import MODELS as MMCV_MODELS
from mmcv.utils import Registry

MODELS = Registry('models', parent=MMCV_MODELS)

ALGORITHMS = MODELS
MUTABLES = MODELS
DISTILLERS = MODELS
LOSSES = MODELS
OPS = MODELS
PRUNERS = MODELS
QUANTIZERS = MODELS
ARCHITECTURES = MODELS
MUTATORS = MODELS


def build_algorithm(cfg):
    """Build compressor."""
    return ALGORITHMS.build(cfg)


def build_architecture(cfg):
    """Build architecture."""
    return ARCHITECTURES.build(cfg)


def build_mutator(cfg):
    """Build mutator."""
    return MUTATORS.build(cfg)


def build_distiller(cfg):
    """Build distiller."""
    return DISTILLERS.build(cfg)


def build_pruner(cfg):
    """Build pruner."""
    return PRUNERS.build(cfg)


def build_mutable(cfg):
    """Build mutable."""
    return MUTABLES.build(cfg)


def build_op(cfg):
    """Build op."""

    # TODO just example
    args_mapping = getattr(cfg, 'args_mapping', None)
    if args_mapping:
        cfg_ = deepcopy(cfg)
        for origin_key, mapping_key in args_mapping.items():
            val = cfg_.pop(origin_key)
            assert mapping_key not in cfg_
            cfg_[mapping_key] = val
    else:
        cfg_ = cfg

    return OPS.build(cfg_)


def build_loss(cfg):
    """Build loss."""
    return LOSSES.build(cfg)
