#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file  : inference.py
@author: zijun
@contact : zijun_sun@shannonai.com
@date  : 2020/11/11 19:56
@version: 1.0
@desc  : 
"""
import argparse
from functools import partial

import torch
from pytorch_lightning import LightningModule, Trainer
from torch.nn import functional as F, CrossEntropyLoss
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer
from transformers.modeling_roberta import RobertaForSequenceClassification

from datasets.collate_functions import collate_to_max_length
from datasets.semi_roberta_dataset import SemiRobertaDataset
from metrics.classification import Accuracy


class RobertaClassificationModel(LightningModule):

    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.args = args
        self.tokenizer = RobertaTokenizer.from_pretrained(self.args.roberta_path)
        self.model = RobertaForSequenceClassification.from_pretrained(self.args.roberta_path)
        self.loss_fn = CrossEntropyLoss()
        self.metric = Accuracy(num_classes=2)
        self.num_gpus = len(str(self.args.gpus).split(","))

    def forward(self, inputs_ids):
        y_hat = self.model(inputs_ids)
        return y_hat[0]

    def get_dataloader(self, directory, prefix) -> DataLoader:
        """构造统一的dataloader方法"""
        dataset = SemiRobertaDataset(directory=directory, prefix=prefix)

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            collate_fn=partial(collate_to_max_length, fill_values=[1, -100])
        )

        return dataloader

    def compute_loss(self, y_hat, y):
        """loss计算函数"""
        loss = self.loss_fn(y_hat, y)
        return loss

    def compute_metric(self, y_hat, y):
        """
            计算准确率的函数
        Args:
            y_hat: 模型预测的y_hat
            y: 数据真实标签y
        """
        predict_scores = F.softmax(y_hat, dim=1)
        predict_labels = torch.argmax(predict_scores, dim=-1)
        acc = self.metric(predict_labels, y)
        return acc

    def test_dataloader(self) -> DataLoader:
        """validation的dataloader"""
        directory = self.args.imdb_data_path
        return self.get_dataloader(directory=directory, prefix="test")

    def test_step(self, batch, batch_idx):
        """对single batch进行validation"""
        inputs_ids, label = batch
        y_hat = self(inputs_ids)
        loss = F.cross_entropy(y_hat, label.view(-1))
        acc = self.compute_metric(y_hat, label.view(-1))
        return {'test_loss': loss, 'test_acc': acc}

    def test_epoch_end(self, outputs):
        """对all batch进行validation"""
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        acc = torch.stack([x['test_acc'].float() for x in outputs]).mean()
        acc = acc / self.num_gpus
        tensorboard_logs = {'test_loss': avg_loss, 'test_acc': acc}
        return {'test_loss': avg_loss, 'log': tensorboard_logs}


def add_model_specific_args():
    parser = argparse.ArgumentParser(description="inference")
    parser.add_argument("--imdb_data_path", type=str, help="root_path")
    parser.add_argument("--roberta_path", required=True, type=str, help="path to save checkpoint and logs")
    parser.add_argument("--checkpoint_path", required=True, type=str, help="path to save checkpoint and logs")
    parser.add_argument("--batch_size", required=True, type=int, help="path to save checkpoint and logs")
    return parser


def inference():
    parser = add_model_specific_args()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    model = RobertaClassificationModel(args)
    checkpoint = torch.load(args.checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])
    trainer = Trainer.from_argparse_args(args,
                                         distributed_backend="ddp")

    trainer.test(model)


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()

    inference()
