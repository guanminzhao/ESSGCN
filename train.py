# -*- coding: utf-8 -*-
# file: train.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.
import copy
import logging
import argparse
import math
import os
import sys
import random
import numpy
import fitlog

from sklearn import metrics
from time import strftime, localtime

from transformers import BertModel, RobertaModel

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from data_utils import build_tokenizer, build_embedding_matrix, Tokenizer4Bert, ABSADataset
from ESSGCN import ESSGCN
from losses import SupConLoss, Syntactic_Reliability_ConLoss

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))
Bert_Path = './bert'

class Instructor:
    def __init__(self, opt):
        self.opt = opt
        if 'essgcn' in opt.model_name:
            tokenizer = Tokenizer4Bert(opt.max_seq_len, opt.pretrained_bert_name)
            bert = BertModel.from_pretrained(Bert_Path)
            self.model = opt.model_class(bert, opt).to(opt.device)
        else:
            tokenizer = build_tokenizer(
                fnames=[opt.dataset_file['train'], opt.dataset_file['test']],
                max_seq_len=opt.max_seq_len,
                dat_fname='{0}_tokenizer.dat'.format(opt.dataset))
            embedding_matrix = build_embedding_matrix(
                word2idx=tokenizer.word2idx,
                embed_dim=opt.embed_dim,
                dat_fname='{0}_{1}_embedding_matrix.dat'.format(str(opt.embed_dim), opt.dataset))
            self.model = opt.model_class(embedding_matrix, opt).to(opt.device)

        self.trainset = ABSADataset(opt.dataset_file['train'], tokenizer)
        self.testset = ABSADataset(opt.dataset_file['test'], tokenizer)
        assert 0 <= opt.valset_ratio < 1
        if opt.valset_ratio > 0:
            valset_len = int(len(self.trainset) * opt.valset_ratio)
            self.trainset, self.valset = random_split(self.trainset, (len(self.trainset)-valset_len, valset_len))
        else:
            self.valset = self.testset

        if opt.device.type == 'cuda':
            logger.info('cuda memory allocated: {}'.format(torch.cuda.memory_allocated(device=opt.device.index)))
        self._print_args()

    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        logger.info('> n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        logger.info('> training arguments:')
        for arg in vars(self.opt):
            logger.info('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))

    def _reset_params(self):
        for child in self.model.children():
            if type(child) != BertModel:  # skip bert params
                for p in child.parameters():
                    if p.requires_grad:
                        if len(p.shape) > 1:
                            self.opt.initializer(p)
                        else:
                            stdv = 1. / math.sqrt(p.shape[0])
                            torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    def _train(self, criterion, criterion_cl1, criterion_cl2, optimizer, train_data_loader, val_data_loader):
        max_val_acc = 0
        max_val_f1 = 0
        max_val_epoch = 0
        global_step = 0
        path = None
        for i_epoch in range(self.opt.num_epoch):
            logger.info('>' * 100)
            logger.info('epoch: {}'.format(i_epoch))
            n_correct, n_total, loss_total, sem_loss_cl_total, syn_loss_cl_total = 0, 0, 0, 0, 0
            penal_total, loss_LER_total, loss_cl3_total = 0, 0, 0
            # switch model to training mode
            self.model.train()
            for i_batch, batch in enumerate(train_data_loader):
                global_step += 1
                # clear gradient accumulators
                optimizer.zero_grad()

                inputs = [batch[col].to(self.opt.device) for col in self.opt.inputs_cols]
                outputs, h_sem, h_syn, penal = self.model(inputs)
                targets = batch['polarity'].to(self.opt.device)
                cl_targets = batch['cl_polarity'].unsqueeze(dim=0).to(self.opt.device)
                loss_LER  = torch.exp(1.0 + torch.cosine_similarity(self.model.L0, self.model.L1))-1.0
                loss = criterion(outputs, targets) + self.opt.alpha * penal + self.opt.beta * loss_LER
                loss_cl1 = criterion_cl1(h_sem.unsqueeze(dim=1), targets)
                loss_cl2 = criterion_cl1(h_syn.unsqueeze(dim=1), targets)
                loss_cl3 = criterion_cl2(h_syn.unsqueeze(dim=1), torch.cat((self.model.L0, self.model.L1), dim=0), cl_targets)

                loss = loss + self.opt.gamma * (loss_cl1 + loss_cl2) + self.opt.delta * loss_cl3
                loss.backward()
                optimizer.step()

                n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                n_total += len(outputs)
                sem_loss_cl_total += loss_cl1 * len(outputs)
                syn_loss_cl_total += loss_cl2 * len(outputs)
                penal_total += penal * len(outputs)
                loss_LER_total += loss_LER.item() * len(outputs)
                loss_cl3_total += loss_cl3 * len(outputs)
                loss_total += loss.item() * len(outputs)
                if global_step % self.opt.log_step == 0:
                    train_acc = n_correct / n_total
                    train_loss = loss_total / n_total
                    sem_train_cl_loss = sem_loss_cl_total / n_total
                    syn_train_cl_loss = syn_loss_cl_total / n_total
                    penal_train = penal_total / n_total
                    loss_LER_train = loss_LER_total / n_total
                    loss_cl3_train = loss_cl3_total / n_total



                    val_acc, val_f1 = self._evaluate_acc_f1(val_data_loader)

                    logger.info('loss: {:.4f},sem_loss_cl:{:.4f}, syn_loss_cl:{:.4f},penal:{:.4f},loss_LER:{:.4f},loss_cl:{:.4f} '
                                'train_acc: {:.4f} val_acc: {:.4f}, val_f1: {:.4f}'.format(train_loss, sem_train_cl_loss
                                , syn_train_cl_loss, penal_train, loss_LER_train, loss_cl3_train,  train_acc, val_acc, val_f1))



                    if val_acc > max_val_acc:
                        max_val_acc = val_acc
                        max_val_epoch = i_epoch
                        max_val_f1 = val_f1
                        if not os.path.exists('state_dict'):
                            os.mkdir('state_dict')
                        path = 'state_dict/{0}_{1}_val_acc_{2}_val_f1_{3}'.format(self.opt.model_name, self.opt.dataset, round(val_acc, 4), round(val_f1, 4))
                        self.best_model = copy.deepcopy(self.model)
                        # torch.save(self.model.state_dict(), path)
                        logger.info('>> saved: {}'.format(path))
                    if val_acc == max_val_acc and val_f1 > max_val_f1:
                        max_val_f1 = val_f1
                        max_val_acc = val_acc
                        max_val_epoch = i_epoch
                        if not os.path.exists('state_dict'):
                            os.mkdir('state_dict')
                        path = 'state_dict/{0}_{1}_val_acc_{2}_val_f1_{3}'.format(self.opt.model_name, self.opt.dataset,
                                                                                  round(val_acc, 4), round(val_f1, 4))
                        self.best_model = copy.deepcopy(self.model)
                        logger.info('>> saved: {}'.format(path))
            if i_epoch - max_val_epoch >= self.opt.patience:
                print('>> early stop.')
                break
        return path

    def _evaluate_acc_f1(self, data_loader):
        n_correct, n_total = 0, 0
        t_targets_all, t_outputs_all = None, None
        # switch model to evaluation mode
        self.model.eval()
        with torch.no_grad():
            for i_batch, t_batch in enumerate(data_loader):
                t_inputs = [t_batch[col].to(self.opt.device) for col in self.opt.inputs_cols]
                t_targets = t_batch['polarity'].to(self.opt.device)
                t_outputs, _, _, _ = self.model(t_inputs)
                # t_outputs = self.model(t_inputs)

                n_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().item()
                n_total += len(t_outputs)

                if t_targets_all is None:
                    t_targets_all = t_targets
                    t_outputs_all = t_outputs
                else:
                    t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0)

        acc = n_correct / n_total
        f1 = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 1, 2], average='macro')
        return acc, f1

    def run(self):
        # Loss and Optimizer
        criterion = nn.CrossEntropyLoss()
        criterion_cl1 = SupConLoss()
        criterion_cl2 = Syntactic_Reliability_ConLoss()
        _params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = self.opt.optimizer(_params, lr=self.opt.lr, weight_decay=self.opt.l2reg)

        train_data_loader = DataLoader(dataset=self.trainset, batch_size=self.opt.batch_size, shuffle=True)
        test_data_loader = DataLoader(dataset=self.testset, batch_size=self.opt.batch_size, shuffle=False)
        val_data_loader = DataLoader(dataset=self.valset, batch_size=self.opt.batch_size, shuffle=False)

        self._reset_params()
        best_model_path = self._train(criterion, criterion_cl1, criterion_cl2, optimizer, train_data_loader, val_data_loader)
        torch.save(self.best_model.state_dict(), best_model_path)
        self.model.load_state_dict(torch.load(best_model_path))
        test_acc, test_f1 = self._evaluate_acc_f1(test_data_loader)
        logger.info('>> test_acc: {:.4f}, test_f1: {:.4f}'.format(test_acc, test_f1))


def main():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='essgcn', type=str)
    parser.add_argument('--datasets', default='restaurant', type=str, help='twitter, restaurant, laptop')
    parser.add_argument('--k', default=0.01, type=float, help='around 0.01~0.001')
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--lr', default=2e-5, type=float, help='try 5e-5, 2e-5 for BERT, 1e-3 for others')
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--ffn_dropout', default=0.3, type=float)
    parser.add_argument('--gat_dropout', default=0.5, type=float)
    parser.add_argument('--l2reg', default=0.01, type=float)
    parser.add_argument('--num_epoch', default=20, type=int, help='try larger number for non-BERT models')
    parser.add_argument('--batch_size', default=16, type=int, help='try 16, 32, 64 for BERT models')
    parser.add_argument('--log_step', default=10, type=int)
    parser.add_argument('--embed_dim', default=300, type=int)
    parser.add_argument('--hidden_dim', default=300, type=int)
    parser.add_argument('--bert_dim', default=768, type=int)
    parser.add_argument('--pretrained_bert_name', default='bert-base-uncased', type=str, help='roberta-base, bert-base-uncased')
    parser.add_argument('--max_seq_len', default=110, type=int)
    parser.add_argument('--polarities_dim', default=3, type=int)
    parser.add_argument('--hops', default=3, type=int)
    parser.add_argument('--patience', default=5, type=int)
    parser.add_argument('--device', default=None, type=str, help='e.g. cuda:0')
    parser.add_argument('--alpha', default=1, type=float)
    parser.add_argument('--beta', default=1, type=float)
    parser.add_argument('--gamma', default=1, type=float)
    parser.add_argument('--delta', default=1, type=float)
    parser.add_argument('--num', default='(8,9,10,11)', type=str)
    parser.add_argument('--noise_lambda', default=0.6, type=float)
    parser.add_argument('--sign', default='1', type=str)
    parser.add_argument('--seed', default=1342082, type=int, help='set seed for reproducibility')
    parser.add_argument('--valset_ratio', default=0, type=float, help='set ratio between 0 and 1 for validation support')
    parser.add_argument('--local_context_focus', default='cdm', type=str, help='local context focus mode, cdw or cdm')
    parser.add_argument('--SRD', default=3, type=int, help='semantic-relative-distance, see the paper of LCF-BERT model')


    opt = parser.parse_args()


    if opt.seed is not None:
        random.seed(opt.seed)
        numpy.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(opt.seed)

    model_classes = {
        'essgcn': ESSGCN,
    }
    dataset_files = {
        'twitter': {
            'train': './datasets/cl_data/Tweet.raw',
            'test': './datasets/acl-14-short-data/test.raw'
        },
        'restaurant': {
            'train': './datasets/cl_data/Restaurants_Train.xml.seg',
            'test': './datasets/semeval14/Restaurants_Test_Gold.xml.seg'
        },
        'laptop': {
            'train': './datasets/cl_data/Laptops_Train.xml.seg',
            'test': './datasets/semeval14/Laptops_Test_Gold.xml.seg'
        }
    }
    input_colses = {
        'essgcn': ['concat_bert_indices', 'concat_segments_indices', 'dependency_graph', 'new_graph', 'aspect_mask'],

    }
    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal_,
        'orthogonal_': torch.nn.init.orthogonal_,
    }
    optimizers = {
        'adadelta': torch.optim.Adadelta,  # default lr=1.0
        'adagrad': torch.optim.Adagrad,  # default lr=0.01
        'adam': torch.optim.Adam,  # default lr=0.001
        'adamax': torch.optim.Adamax,  # default lr=0.002
        'asgd': torch.optim.ASGD,  # default lr=0.01
        'rmsprop': torch.optim.RMSprop,  # default lr=0.01
        'sgd': torch.optim.SGD,
        'adamw': torch.optim.AdamW
    }
    opt.model_class = model_classes[opt.model_name]
    opt.dataset_file = dataset_files[opt.dataset]
    opt.inputs_cols = input_colses[opt.model_name]
    opt.initializer = initializers[opt.initializer]
    opt.optimizer = optimizers[opt.optimizer]
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if opt.device is None else torch.device(opt.device)

    log_file = '{}-{}-{}.log'.format(opt.model_name, opt.dataset, strftime("%y%m%d-%H%M", localtime()))
    logger.addHandler(logging.FileHandler(log_file))

    ins = Instructor(opt)
    ins.run()

if __name__ == '__main__':
    main()
