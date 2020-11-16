#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file  : roberta_dataset.py
@author: zijun
@contact : zijun_sun@shannonai.com
@date  : 2020/10/22 21:13
@version: 1.0
@desc  : 
"""

import os

import torch
from shannon_preprocessor.mmap_dataset import MMapIndexedDataset
from torch.utils.data import Dataset
from transformers import RobertaTokenizer


class RobertaMaskedLMDataset(Dataset):
    """Dynamic Masked Language Model Dataset"""

    def __init__(self, directory, prefix, fields=None, roberta_base_path: str = "", mask_prob: float = 0.15,
                 max_length: int = 512, use_memory=False):
        super().__init__()
        fields = fields or ["input_ids"]
        self.fields2datasets = {}
        self.fields = fields
        self.mask_prob = mask_prob
        self.max_length = max_length

        self.tokenizer = RobertaTokenizer.from_pretrained(roberta_base_path)

        self.cls, self.sep = self.tokenizer.cls_token_id, self.tokenizer.sep_token_id

        for field in fields:
            self.fields2datasets[field] = MMapIndexedDataset(os.path.join(directory, f"{prefix}.{field}"),
                                                             use_memory=use_memory)

    def __len__(self):
        return len(self.fields2datasets[self.fields[0]])

    def __getitem__(self, item):
        input_ids = self.fields2datasets["input_ids"][item]

        masked_indices = self.char_mask(input_ids)

        labels = input_ids.clone()

        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        input_ids[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return input_ids, labels

    def char_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        random mask chars
        Args:
            input_ids: input ids [sent_len]
        Returns:
            masked_indices:[sent_len], if True, mask this token
        """
        probability_matrix = torch.full(input_ids.shape, self.mask_prob)
        special_tokens_mask = self.tokenizer.get_special_tokens_mask(input_ids.tolist(),
                                                                     already_has_special_tokens=True)
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        if self.tokenizer._pad_token is not None:
            padding_mask = input_ids.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        return masked_indices


def unit_test():
    roberta_base = "/data/nfsdata2/sunzijun/loop/roberta-base"
    data_path = "/data/nfsdata2/sunzijun/loop/imdb/roberta_test/bin"

    tokenizer = RobertaTokenizer.from_pretrained(roberta_base)
    prefix = "dev"

    dataset = RobertaMaskedLMDataset(data_path, roberta_base_path=roberta_base,
                                     prefix=prefix, max_length=512, fields=["input_ids"])
    print(len(dataset))
    from tqdm import tqdm
    for d in tqdm(dataset):
        print([v.shape for v in d])
        print(tokenizer.decode(d[0].tolist(), skip_special_tokens=False))
        tgt = [src if label == -100 else label for src, label in zip(d[0].tolist(), d[1].tolist())]
        print(tokenizer.decode(tgt, skip_special_tokens=False))


if __name__ == '__main__':
    unit_test()
