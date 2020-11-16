#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file  : roberta_tokenize.py
@author: zijun
@contact : zijun_sun@shannonai.com
@date  : 2020/10/22 21:13
@version: 1.0
@desc  : 
"""

from transformers import RobertaTokenizer


import os
import numpy as np
from argparse import ArgumentParser
from typing import Dict, List

import torch
from shannon_preprocessor.dataset_reader import DatasetReader

os.environ["TOKENIZERS_PARALLELISM"] = "false"


@DatasetReader.register("roberta_tokenize")
class RobertaTokenizeReader(DatasetReader):
    """
    process pretrain data
        1. pack sentence to max length
        2. bert tokenize
    todo
    """

    def __init__(self, args):
        super().__init__(args)
        print("args: ", args)
        self.max_len = args.max_len
        self.tokenizer = RobertaTokenizer.from_pretrained(args.roberta_base)
        self.prev_tokens = []

    @staticmethod
    def add_args(parser: ArgumentParser):
        """Add specific arguments to the dataset reader."""
        parser.add_argument("--max_len", type=int, default=512)
        parser.add_argument("--roberta_base", required=True, type=str)

    @property
    def fields2dtypes(self):
        """
        define numpy dtypes of each field.
        """
        dic = {
            "input_ids": np.uint16,
        }
        return dic

    def get_inputs(self, line: str) -> List[Dict[str, torch.Tensor]]:
        """get input from file"""
        sent = line.strip()
        output = []

        roberta_tokens = self.tokenizer.encode(sent, add_special_tokens=False)

        if len(roberta_tokens) > self.max_len - 2:
            roberta_tokens = roberta_tokens[:self.max_len - 2]

        if len(roberta_tokens) + len(self.prev_tokens) > self.max_len - 2:
            input_ids = [0] + self.prev_tokens + [2]
            assert len(input_ids) <= self.max_len
            output.append({"input_ids": torch.LongTensor(input_ids)})
            self.prev_tokens = roberta_tokens
        else:
            self.prev_tokens += roberta_tokens

        return output


def run_roberta_tokenize_reader():
    class Args:
        max_len = 128
        roberta_base = '/data/nfsdata2/sunzijun/loop/roberta-base'
        input_file = "/data/nfsdata2/sunzijun/loop/imdb/reviews.txt"

    reader = RobertaTokenizeReader(Args)
    with open(Args.input_file) as fin:
        for line in fin:
            try:
                print(line.strip())
                y = reader.get_inputs(line)
                print(y)
            except Exception as e:
                print(f"Error on {y}")
                continue


if __name__ == '__main__':
    run_roberta_tokenize_reader()
