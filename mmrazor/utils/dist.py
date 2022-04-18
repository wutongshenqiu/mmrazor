# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.runner.dist_utils import master_only


@master_only
def master_only_print(*args, **kwargs) -> None:
    print(*args, **kwargs)
