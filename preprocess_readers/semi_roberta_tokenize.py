#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file  : semi_roberta_tokenize.py
@author: zijun
@contact : zijun_sun@shannonai.com
@date  : 2020/10/30 16:12
@version: 1.0
@desc  : 
"""

import os
from argparse import ArgumentParser
from typing import Dict, List

import numpy as np
import torch
from shannon_preprocessor.dataset_reader import DatasetReader
from transformers import RobertaTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"


@DatasetReader.register("semi_roberta_tokenize")
class SemiRobertaTokenizeReader(DatasetReader):

    def __init__(self, args):
        super().__init__(args)
        print("args: ", args)
        self.max_len = args.max_len
        self.tokenizer = RobertaTokenizer.from_pretrained(args.roberta_base)

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
            "label": np.uint16,
        }
        return dic

    def get_inputs(self, line: str) -> List[Dict[str, torch.Tensor]]:
        """get input from file"""
        output = []
        # remove short sentence
        if len(line) <= 3 or line[1] != '\t':
            assert 1 == 0
        label, sentence = line.strip().split('\t', 1)
        input_ids = self.tokenizer.encode(sentence, add_special_tokens=False)
        # cut longer sentence to max length
        if len(input_ids) > self.max_len - 2:
            input_ids = input_ids[:self.max_len - 2]
        input_ids = [0] + input_ids + [2]

        assert len(input_ids) <= self.max_len
        assert len(input_ids) > 0
        # construct result
        output.append({
            'input_ids': torch.LongTensor(input_ids),
            'label': torch.LongTensor([int(label)])
        })
        return output


def run_roberta_tokenize_reader():
    class Args:
        max_len = 128
        roberta_base = '/data/nfsdata2/sunzijun/loop/roberta-base'
        input_file = "/data/nfsdata2/sunzijun/loop/experiments/train_25k/teacher_data/train.txt"

    reader = SemiRobertaTokenizeReader(Args)
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
