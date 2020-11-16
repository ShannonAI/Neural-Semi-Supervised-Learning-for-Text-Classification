# encoding: utf-8
"""
@author: zijun
@contact : zijun_sun@shannonai.com

@version: 1.0
@file: accuracy
@time: 2020/7/11 11:51

"""

from typing import Any, Optional

import torch
from pytorch_lightning.metrics.functional.classification import (
    accuracy
)
from pytorch_lightning.metrics.metric import TensorMetric


class Accuracy(TensorMetric):
    """
    Computes the accuracy classification score
    Example:
        >>> pred = torch.tensor([0, 1, 2, 3])
        >>> target = torch.tensor([0, 1, 2, 2])
        >>> metric = Accuracy(num_classes=4)
        >>> metric(pred, target)
        tensor(1.)
    """

    def __init__(
        self,
        num_classes: Optional[int] = None,
        reduce_group: Any = None,
        reduce_op: Any = None,
    ):
        """
        Args:
            num_classes: number of classes
            reduction: a method for reducing accuracies over labels (default: takes the mean)
                Available reduction methods:
                - elementwise_mean: takes the mean
                - none: pass array
                - sum: add elements
            reduce_group: the process group to reduce metric results from DDP
            reduce_op: the operation to perform for ddp reduction
        """
        super().__init__(name='accuracy',
                         reduce_group=reduce_group,
                         reduce_op=reduce_op)
        self.num_classes = num_classes

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Actual metric computation
        Args:
            pred: predicted labels
            target: ground truth labels
            mask: only calculate metrics where mask==1
        Return:
            A Tensor with the classification score.
        """
        return accuracy(pred=pred, target=target, num_classes=self.num_classes)


def main():
    pred = torch.tensor([0, 1, 2, 3])
    target = torch.tensor([0, 1, 2, 2])
    metric = Accuracy(num_classes=4)
    print(metric(pred, target))


if __name__ == '__main__':
    main()
