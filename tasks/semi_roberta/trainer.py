#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file  : trainer.py
@author: zijun
@contact : zijun_sun@shannonai.com
@date  : 2020/10/29 17:56
@version: 1.0
@desc  :
"""

import argparse
import json
import os
import shutil
import uuid
from functools import partial

import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.nn import functional as F, CrossEntropyLoss
from torch.utils.data import DataLoader
from transformers import AdamW, RobertaConfig
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
        # self.model = RobertaForSequenceClassification.from_pretrained(self.args.roberta_path)

        self.robert_config = RobertaConfig.from_pretrained(self.args.roberta_path, output_hidden_states=False)
        self.model = RobertaForSequenceClassification(self.robert_config)

        self.loss_fn = CrossEntropyLoss()
        self.metric = Accuracy(num_classes=2)
        gpus_string = self.args.gpus if not self.args.gpus.endswith(',') else self.args.gpus[:-1]
        self.num_gpus = len(gpus_string.split(","))

        self.predict_neg = []
        self.predict_pos = []

    def forward(self, inputs_ids):
        y_hat = self.model(inputs_ids)
        return y_hat[0]

    def get_dataloader(self, directory, prefix) -> DataLoader:
        """construct unified dataloader"""
        dataset = SemiRobertaDataset(directory=directory, prefix=prefix)

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.workers,
            shuffle=True,
            collate_fn=partial(collate_to_max_length, fill_values=[1, -100])
        )

        return dataloader

    def train_dataloader(self) -> DataLoader:
        """dataloader for training"""
        if self.args.mode == 'train_teacher' or self.args.mode == 'fine_tune':
            directory = self.args.imdb_data_path
        elif self.args.mode == 'train_student':
            directory = self.args.student_data_path
        return self.get_dataloader(directory=directory, prefix="train")

    def training_step(self, batch, batch_idx):
        inputs_ids, label = batch
        y_hat = self.forward(inputs_ids)
        loss = self.compute_loss(y_hat, label.view(-1))
        acc = self.compute_metric(y_hat, label.view(-1))
        tensorboard_logs = {'train_loss': loss, 'train_acc': acc,
                            "lr": self.trainer.optimizers[0].param_groups[0]['lr']}
        return {'loss': loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          betas=(0.9, 0.98),  # according to RoBERTa paper
                          lr=self.args.lr,
                          eps=self.args.adam_epsilon)
        num_gpus = len(str(self.args.gpus).split(","))
        t_total = len(self.train_dataloader()) * self.args.max_epochs // (
            self.args.accumulate_grad_batches * num_gpus) + 1
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=self.args.lr, pct_start=float(self.args.warmup_steps / t_total),
            total_steps=t_total, anneal_strategy='linear'
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def compute_loss(self, y_hat, y):
        """loss function"""
        loss = self.loss_fn(y_hat, y)
        return loss

    def compute_metric(self, y_hat, y):
        """
            calculate accuracy
        Args:
            y_hat: model output, y_hat
            y: ground truth label, y
        """
        predict_scores = F.softmax(y_hat, dim=1)
        predict_labels = torch.argmax(predict_scores, dim=-1)
        acc = self.metric(predict_labels, y)
        return acc

    def val_dataloader(self) -> DataLoader:
        """validation dataloader"""
        directory = self.args.imdb_data_path
        return self.get_dataloader(directory=directory, prefix="test")

    def validation_step(self, batch, batch_idx):
        inputs_ids, label = batch
        y_hat = self(inputs_ids)
        loss = F.cross_entropy(y_hat, label.view(-1))
        acc = self.compute_metric(y_hat, label.view(-1))
        return {'val_loss': loss, 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'].float() for x in outputs]).mean() / self.num_gpus
        tensorboard_logs = {'val_loss': avg_loss, 'val_acc': avg_acc}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def test_dataloader(self) -> DataLoader:
        """dataloader for label unlabeled data"""
        return self.get_dataloader(directory=self.args.semi_data_path, prefix="train")

    def test_step(self, batch, batch_idx):
        """process unlabeled data"""
        inputs_ids, label = batch
        y_hat = self(inputs_ids)
        scores = F.softmax(y_hat)
        labels = torch.argmax(scores, dim=1)
        # convert result
        for label, score, input_ids in zip(labels.tolist(), scores.tolist(), inputs_ids.tolist()):
            sentence = self.tokenizer.decode(input_ids, skip_special_tokens=True).strip()
            if label == 0:
                self.predict_neg.append(str(score[0]) + '\t' + sentence + '\n')
            elif label == 1:
                self.predict_pos.append(str(score[1]) + '\t' + sentence + '\n')
        return {'loss': 1}

    def test_epoch_end(self, outputs):
        """write the label data to file"""
        pre_name = str(uuid.uuid1())
        # write pos samples to file
        pos_path = os.path.join(self.args.save_path, 'predictions', pre_name + '-pos.txt')
        with open(pos_path, 'w', encoding='utf8') as fout:
            fout.writelines(self.predict_pos)
        # write neg samples to file
        neg_path = os.path.join(self.args.save_path, 'predictions', pre_name + '-neg.txt')
        with open(neg_path, 'w', encoding='utf8') as fout:
            fout.writelines(self.predict_neg)
        return {'loss': 1}


def find_best_checkpoint(path: str):
    checkpoints = []
    for file in os.listdir(path):
        if file.__contains__('tmp') or file.__contains__('.txt') or file.__contains__('.txt'):
            continue
        acc = float(file.split('=')[-1].replace(".ckpt", ""))
        checkpoints.append((acc, file))
    orderd_checkpoints = sorted(checkpoints, key=lambda k: k[0], reverse=True)
    bert_checkpoint = os.path.join(path, orderd_checkpoints[0][1])

    return bert_checkpoint


def add_model_specific_args():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--mode", required=True, type=str, help='train mode')

    parser.add_argument("--max_length", type=int, default=512, help="sentence max length（after tokenize）")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="warmup steps")
    parser.add_argument("--batch_size", type=int, default=10, help="batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="learning rate")
    parser.add_argument("--workers", type=int, default=0, help="num workers for dataloader")
    parser.add_argument("--patience", default=5, type=int, help="patience to early stop")

    parser.add_argument("--roberta_path", required=True, type=str, help="roberta base dir")
    parser.add_argument("--imdb_data_path", type=str, help="imdb data bin path")
    parser.add_argument("--student_data_path", type=str, help="data path for train student model")
    parser.add_argument("--reviews_data_path", type=str, help="reviews data path")
    parser.add_argument("--save_path", required=True, type=str, help="path to save checkpoint and logs")
    parser.add_argument("--best_teacher_checkpoint_path", type=str, help="best teacher checkpoint path")

    return parser


def train_model(args):
    # generate dir name
    checkpoint_name = args.mode + '_checkpoint'
    log_name = args.mode + '_log'
    checkpoint_path = os.path.join(args.save_path, checkpoint_name)
    args.checkpoint_path = checkpoint_path
    log_path = os.path.join(args.save_path, log_name)
    # init model
    model = RobertaClassificationModel(args)
    if args.mode == 'fine_tune':
        best_student_checkpoint = find_best_checkpoint(os.path.join(args.save_path, 'train_student_checkpoint'))
        checkpoint = torch.load(best_student_checkpoint, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['state_dict'])
    # init model env
    if os.path.exists(checkpoint_path):
        shutil.rmtree(checkpoint_path)
    if os.path.exists(log_path):
        shutil.rmtree(log_path)
    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(checkpoint_path, '{epoch}-{val_loss:.4f}-{val_acc:.4f}'),
        save_top_k=5,
        monitor="val_acc",
        mode='max',
        period=1
    )
    logger = TensorBoardLogger(
        save_dir=args.save_path,
        name=log_name
    )
    early_stop = EarlyStopping(
        monitor='val_acc',
        patience=args.patience,
        strict=False,
        verbose=False,
        mode='max'
    )
    # save args
    with open(os.path.join(checkpoint_path, "args.json"), 'w') as f:
        args_dict = args.__dict__
        del args_dict['tpu_cores']
        json.dump(args_dict, f, indent=4)

    trainer = Trainer.from_argparse_args(args,
                                         checkpoint_callback=checkpoint_callback,
                                         early_stop_callback=early_stop,
                                         logger=logger,
                                         distributed_backend="ddp")

    trainer.fit(model)


def label_unlabeled_data(args):
    # clear previoous prediction dir
    prediction_path = os.path.join(args.save_path, 'predictions')
    if os.path.exists(prediction_path):
        shutil.rmtree(prediction_path)
    os.mkdir(prediction_path)

    model = RobertaClassificationModel(args)

    checkpoint_path = args.best_teacher_checkpoint_path
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])

    trainer = Trainer(gpus=args.gpus)
    trainer.test(model)

    # read all prediction files
    all_pos = []
    all_neg = []
    for file in os.listdir(prediction_path):
        if file in ['pos.txt', 'neg.txt']:
            continue
        predict_file = os.path.join(prediction_path, file)
        with open(predict_file, 'r', encoding='utf8') as f:
            lines = f.readlines()
        if file.__contains__('pos'):
            all_pos.extend(lines)
        elif file.__contains__('neg'):
            all_neg.extend(lines)
        os.remove(predict_file)

    # merge all file into one file
    with open(os.path.join(prediction_path, 'pos.txt'), 'w', encoding='utf8') as f:
        f.writelines(all_pos)
    with open(os.path.join(prediction_path, 'neg.txt'), 'w', encoding='utf8') as f:
        f.writelines(all_neg)


def main():
    parser = add_model_specific_args()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    if args.mode == 'label':
        label_unlabeled_data(args)
    else:
        train_model(args)


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()

    main()
