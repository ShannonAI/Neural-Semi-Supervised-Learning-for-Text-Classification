#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file  : random_seed.py
@author: zijun
@contact : zijun_sun@shannonai.com
@date  : 2020/11/9 16:57
@version: 1.0
@desc  : 
"""
import numpy as np
import torch


def set_random_seed(seed: int):
    """set seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    # without this line, x would be different in every execution.
    set_random_seed(0)

    x = np.random.random()
    print(x)
