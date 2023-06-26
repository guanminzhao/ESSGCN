# -*- coding: utf-8 -*-
# file: BERT_SPC.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2019. All Rights Reserved.
import torch
import torch.nn as nn


class BERT_SPC(nn.Module):
    def __init__(self, bert, opt, pool='max'):
        super(BERT_SPC, self).__init__()
        self.ffn = nn.Sequential(
            nn.Linear(opt.bert_dim, opt.bert_dim),
            nn.ReLU(),
            nn.Dropout(0.3),  # pool = max 时0.3为最好，pool = mean 时0.5最好
            nn.Linear(opt.bert_dim, opt.polarities_dim),
        )
        for p in self.parameters():
            p.requires_grad = False
        self.bert = bert
        self.pool = pool
        self.dropout = nn.Dropout(opt.dropout)


    def forward(self, inputs):
        concat_bert_indices, concat_segments_indices, aspect_mask = inputs[0], inputs[1], inputs[2]
        output = self.bert(input_ids=concat_bert_indices, token_type_ids=concat_segments_indices,
                                  output_attentions=True)
        if self.pool == 'mean':
            aspect_mask = aspect_mask.eq(0).unsqueeze(dim=-1)
            output = output[0].masked_fill(aspect_mask, 0)
            pre = output.sum(dim=1)/aspect_mask.eq(0).sum(dim=1)
        elif self.pool == 'max':
            aspect_mask = aspect_mask.eq(0).unsqueeze(dim=-1)
            output = output[0].masked_fill(aspect_mask, -10000.0)
            pre, _ = output.max(dim=1)
        logits = self.ffn(pre)
        return logits
