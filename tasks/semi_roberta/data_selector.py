#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file  : data_spliter.py
@author: zijun
@contact : zijun_sun@shannonai.com
@date  : 2020/10/31 21:26
@version: 1.0
@desc  : 
"""
import argparse
import os
import random


def main():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--imdb_data_path", required=True, type=str, help='imdb data path')
    parser.add_argument("--save_path", required=True, type=str, help='path to save data')
    args = parser.parse_args()
    with open(os.path.join(args.imdb_data_path, 'train.txt'), 'r', encoding='utf8') as f:
        train_lines = f.readlines()
    samples = get_predictions(os.path.join(args.save_path, 'predictions'))
    U_size = 3.4 * 10 ** 6
    D_up_size = 10 ** 6

    # step 1: random sample U_size data from 3.4M reviews data
    U_samples = random.sample(samples, U_size)
    pos_samples = [(score, '1', sentence) for label, score, sentence in U_samples if label == 1]
    neg_samples = [(score, '0', sentence) for label, score, sentence in U_samples if label == 0]

    pos_samples = sorted(pos_samples, key=lambda k: k[0], reverse=True)
    neg_samples = sorted(neg_samples, key=lambda k: k[0], reverse=True)

    # step 2: select top D_up_size data from U dataset
    topk_pos_samples = pos_samples[:D_up_size // 2]
    topk_neg_samples = neg_samples[:D_up_size // 2]
    D_s = [label + '\t' + sentence for _, label, sentence in topk_pos_samples + topk_neg_samples]

    # setp 3: concat D_up data and D data
    together_path = os.path.join(args.save_path, 'student_data')
    together_data = D_s + train_lines
    random.shuffle(together_data)
    with open(os.path.join(together_path, 'train.txt'), 'w', encoding='utf8') as f:
        f.writelines(together_data)


def get_predictions(path):
    pos_path = os.path.join(path, 'pos.txt')
    neg_path = os.path.join(path, 'neg.txt')

    pos_samples = []
    neg_samples = []
    # load 3.4M data with label
    with open(pos_path, 'r', encoding='utf8') as f:
        pos_lines = f.readlines()
        for line in pos_lines:
            score, sentence = line.split('\t', 1)
            pos_samples.append((1, float(score), sentence))

    with open(neg_path, 'r', encoding='utf8') as f:
        neg_lines = f.readlines()
        for line in neg_lines:
            score, sentence = line.split('\t', 1)
            neg_samples.append((0, float(score), sentence))

    # merge samples
    samples = pos_samples + neg_samples
    return samples


if __name__ == '__main__':
    main()
