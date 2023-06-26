# -*- coding: utf-8 -*-
# file: BERT_SPC.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2019. All Rights Reserved.
import torch
import torch.nn as nn


class RoBERTa(nn.Module):
    def __init__(self, roberta, opt, pool='max'):
        super(RoBERTa, self).__init__()
        self.pool = pool
        self.roberta = roberta
        self.dropout = nn.Dropout(opt.dropout)
        self.dense = nn.Linear(opt.bert_dim, opt.polarities_dim)
        for n in self.parameters():
            n.requires_grad = True


    def forward(self, inputs):
        text_roberta_indices, attention_mask, aspect_mask = inputs[0], inputs[1], inputs[2]

        output = self.roberta(text_roberta_indices.squeeze(dim=1), attention_mask)
        pooled_output = self.dropout(output[0])
        if self.pool == 'mean':
            aspect_mask = aspect_mask.eq(0).permute(0, 2, 1)
            output = pooled_output.masked_fill(aspect_mask, 0)
            pre = output.sum(dim=1)/aspect_mask.eq(0).sum(dim=1)
        elif self.pool == 'max':
            aspect_mask = aspect_mask.eq(0).permute(0, 2, 1)
            output = pooled_output.masked_fill(aspect_mask, -10000.0)
            pre, _ = output.max(dim=1)
        logits = self.dense(pre)
        return logits
