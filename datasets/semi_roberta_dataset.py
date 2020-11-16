#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file  : semi_roberta_dataset.py
@author: zijun
@contact : zijun_sun@shannonai.com
@date  : 2020/10/30 16:31
@version: 1.0
@desc  : DataSet for semi supervised learning
"""

import os
from functools import partial

from shannon_preprocessor.mmap_dataset import MMapIndexedDataset
from torch.utils.data import Dataset, DataLoader

from datasets.collate_functions import collate_to_max_length


class SemiRobertaDataset(Dataset):

    def __init__(self, directory, prefix, fields=None, max_length: int = 512, use_memory=False):
        super().__init__()
        fields = fields or ["input_ids",  "label"]
        self.fields2datasets = {}
        self.fields = fields
        self.max_length = max_length

        for field in fields:
            self.fields2datasets[field] = MMapIndexedDataset(os.path.join(directory, f"{prefix}.{field}"),
                                                             use_memory=use_memory)

    def __len__(self):
        return len(self.fields2datasets[self.fields[0]])

    def __getitem__(self, item):
        input_ids = self.fields2datasets["input_ids"][item]
        label = self.fields2datasets["label"][item]
        return input_ids, label


def unit_test():
    root_path = "/data/nfsdata2/sunzijun/loop/experiments/train_25k/teacher_data/roberta_bin"
    prefix = "train"
    dataset = SemiRobertaDataset(directory=root_path, prefix=prefix)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=10,
        num_workers=0,
        shuffle=False,
        collate_fn=partial(collate_to_max_length, fill_values=[1, -100])
    )
    for input_ids, label in dataloader:
        print(input_ids.shape)
        print(label.view(-1))
        print()


if __name__ == '__main__':
    unit_test()
