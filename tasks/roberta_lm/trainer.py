#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file  : trainer.py
@author: zijun
@contact : zijun_sun@shannonai.com
@date  : 2020/10/22 21:13
@version: 1.0
@desc  : train roberta language model
"""

import argparse
import os
import shutil
from functools import partial

import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from transformers import AdamW, RobertaTokenizer, RobertaConfig
from transformers.modeling_roberta import RobertaForMaskedLM

from datasets.collate_functions import collate_to_max_length
from datasets.roberta_dataset import RobertaMaskedLMDataset
from metrics.classification import MaskedAccuracy
from utils.random_seed import set_random_seed

set_random_seed(0)


class SemiRoberta(pl.LightningModule):
    """"""

    def __init__(
        self,
        args: argparse.Namespace
    ):
        """Initialize a model, tokenizer and config."""
        super().__init__()
        self.args = args
        if isinstance(args, argparse.Namespace):
            self.save_hyperparameters(args)
        self.tokenizer = RobertaTokenizer.from_pretrained(self.args.roberta_path)
        self.model = RobertaForMaskedLM.from_pretrained(self.args.roberta_path)

        self.robert_config = RobertaConfig.from_pretrained(self.args.roberta_path, output_hidden_states=False)
        self.model = RobertaForMaskedLM(self.robert_config)

        self.loss_fn = CrossEntropyLoss(reduction="none")
        self.acc = MaskedAccuracy(num_classes=self.tokenizer.vocab_size)

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

    def forward(self, input_ids):
        """"""
        attention_mask = (input_ids != 1).long()
        return self.model(input_ids, attention_mask=attention_mask)

    def get_dataloader(self, prefix) -> DataLoader:
        """构造统一的dataloader方法"""
        dataset = RobertaMaskedLMDataset(directory=self.args.data_dir,
                                         prefix=prefix,
                                         roberta_base_path=self.args.roberta_path,
                                         max_length=self.args.max_length)

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.workers,
            shuffle=True,
            collate_fn=partial(collate_to_max_length, fill_values=[1, -100])
        )

        return dataloader

    def train_dataloader(self) -> DataLoader:
        return self.get_dataloader("train")

    def training_step(self, batch, batch_idx):
        """"""
        loss, acc = self.compute_loss_and_acc(batch)
        tf_board_logs = {
            "train_loss": loss,
            "train_acc": acc,
            "lr": self.trainer.optimizers[0].param_groups[0]['lr']
        }
        return {'loss': loss, 'log': tf_board_logs}

    def val_dataloader(self):
        return self.get_dataloader("dev")

    def validation_step(self, batch, batch_idx):
        """"""
        loss, acc = self.compute_loss_and_acc(batch)
        return {'val_loss': loss, "val_acc": acc}

    def validation_epoch_end(self, outputs):
        """"""
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss, 'val_acc': avg_acc}
        print(avg_loss, avg_acc)
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def compute_loss_and_acc(self, batch):
        """"""
        epsilon = 1e-10
        masked_lms = batch[1].view(-1)
        outputs = self(
            input_ids=batch[0],
        )
        prediction_scores = outputs[0]
        label_mask = (masked_lms >= 0)
        # remove negative mask
        # masked_lms = torch.where(label_mask, masked_lms, torch.tensor(0, device=self.device, dtype=torch.int64))
        loss = self.loss_fn(prediction_scores.view(-1, self.tokenizer.vocab_size),
                            masked_lms)

        predict_labels = torch.argmax(prediction_scores.view(-1, self.tokenizer.vocab_size), dim=-1)
        acc = self.acc(pred=predict_labels,
                       target=masked_lms,
                       mask=label_mask.long())

        label_mask = label_mask.float()
        loss *= label_mask
        loss = loss.sum() / (label_mask.sum() + epsilon)
        return loss, acc


def main():
    """main"""
    # main parser
    parser = argparse.ArgumentParser(description="Training")

    # for path
    parser.add_argument("--data_dir", required=True, type=str, help="data dirs")
    parser.add_argument("--batch_size", type=int, default=10, help="batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="learning rate")
    parser.add_argument("--workers", type=int, default=0, help="num workers for dataloader")
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--warmup_steps", default=10000, type=int, help="warmup steps")
    parser.add_argument("--use_memory", action="store_true", help="load dataset to memory to accelerate.")

    parser.add_argument("--roberta_path", required=True, type=str, help="roberta base dir")
    parser.add_argument("--max_length", default=512, type=int, help="max length of dataset")
    parser.add_argument("--save_path", required=True, type=str, help="path to save checkpoint and logs")

    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    model = SemiRoberta(args)
    print(model)

    checkpoint_path = os.path.join(args.save_path, "checkpoints")
    log_path = os.path.join(args.save_path, "logs")
    if os.path.exists(checkpoint_path):
        shutil.rmtree(checkpoint_path)
    if os.path.exists(log_path):
        shutil.rmtree(log_path)

    logger = TensorBoardLogger(
        save_dir=args.save_path,
        name="logs"
    )
    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(args.save_path, 'checkpoints', '{epoch}-{val_loss:.4f}-{val_acc:.4f}'),
        save_top_k=10,
        save_last=True,
        verbose=True,
        monitor="train_loss",
        period=-1,
        mode="min",
    )
    trainer = Trainer.from_argparse_args(args,
                                         checkpoint_callback=checkpoint_callback,
                                         logger=logger)

    trainer.fit(model)


def convert_checkpint_to_bin(checkpoint_path, bin_path, mode="cpu"):
    """convert saved checkpoint to huggingface bin format"""
    parser = argparse.ArgumentParser(description="Training")
    parser = SemiRoberta.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    model = SemiRoberta(args)

    checkpoint = torch.load(checkpoint_path, map_location=mode)
    model.load_state_dict(checkpoint['state_dict'])
    if not os.path.exists(bin_path):
        os.mkdir(bin_path)
    model.model.save_pretrained(bin_path)


if __name__ == '__main__':
    # train roberta model
    main()

    # sample code to convert checkpoint
    # checkpoint_path = "/userhome/sunzijun/self-learning/roberta_data/large-checkpoints/epoch=7-val_loss=1.2049-val_acc=2.9290.ckpt"
    # out_bin_path = "/userhome/sunzijun/self-learning/roberta_data/pretrains/semi-roberta-large-7"
    # convert_checkpint_to_bin(checkpoint_path, out_bin_path)
