import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer


def pad_and_truncate(sequence, maxlen, dtype='int64', padding='post', truncating='post', value=0):
    x = (np.ones(maxlen) * value).astype(dtype)
    if truncating == 'pre':
        trunc = sequence[-maxlen:]
    else:
        trunc = sequence[:maxlen]
    trunc = np.asarray(trunc, dtype=dtype)
    if padding == 'post':
        x[:len(trunc)] = trunc
    else:
        x[-len(trunc):] = trunc
    return x



class Tokenizer4Bert:
    def __init__(self, max_seq_len, pretrained_bert_name):
        if 'bert-base-uncased' == pretrained_bert_name:
            self.tokenizer = BertTokenizer.from_pretrained(pretrained_bert_name)

        self.max_seq_len = max_seq_len

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post'):
        sequence = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        return pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating)

    def text_to_sequence4bert(self, text):
        sequence = self.tokenizer(text)['input_ids']
        sequence.pop(0)
        sequence.pop(-1)
        sequence = self.tokenizer.convert_ids_to_tokens(sequence)
        return sequence



class ABSADataset(Dataset):
    def __init__(self, fname, tokenizer):
        fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        lines = fin.readlines()
        fin.close()
        fin = open(fname+'.graph', 'rb')
        idx2graph = pickle.load(fin)
        fin.close()
        fin = open(fname+'.newgraph', 'rb')
        new_graph = pickle.load(fin)
        fin.close()

        all_data = []
        if 'cl_data' in fname:
            for i in range(0, len(lines), 4):
                j = int((i/4)*3)
                text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
                aspect = lines[i + 1].lower().strip()
                polarity = lines[i + 2].strip()
                cl_polarity = int(lines[i + 3].strip())
                text = text_left + ' ' + aspect + ' ' + text_right

                text_indices = tokenizer.text_to_sequence(text_left + " " + aspect + " " + text_right)
                aspect_indices = tokenizer.text_to_sequence(aspect)
                aspect_len = np.sum(aspect_indices != 0)
                text_len = np.sum(text_indices != 0)
                polarity = int(polarity) + 1

                concat_bert_indices = tokenizer.text_to_sequence(
                    '[CLS] ' + text_left + " " + aspect + " " + text_right + ' [SEP] ' + aspect + " [SEP]")
                concat_segments_indices = [0] * (text_len + 2) + [1] * (aspect_len + 1)
                concat_segments_indices = pad_and_truncate(concat_segments_indices, tokenizer.max_seq_len)
                text_bert_indices = tokenizer.text_to_sequence(
                    "[CLS] " + text_left + " " + aspect + " " + text_right + " [SEP]")
                aspect_bert_indices = tokenizer.text_to_sequence("[CLS] " + aspect + " [SEP]")

                aspect_mask = ABSADataset.__get_aspect_mask__(aspect, concat_bert_indices, tokenizer)
                aspect_graph = new_graph[j]['graph']

                text_bert_tokens = []
                text_bert__tokens_ids = []
                for token_id, token in enumerate(text.split()):
                    for bert_token in tokenizer.text_to_sequence4bert(token):
                        text_bert_tokens.append(bert_token)
                        text_bert__tokens_ids.append(token_id)

                bert_token_len = len(tokenizer.text_to_sequence4bert(text))
                dependency_graph = idx2graph[j]
                bert_dependency_graph = np.zeros((bert_token_len, bert_token_len), dtype=float)
                for x in range(bert_token_len):
                    for y in range(bert_token_len):
                        bert_dependency_graph[x][y] = dependency_graph[text_bert__tokens_ids[x]][
                            text_bert__tokens_ids[y]]

                bert_graph = np.zeros(bert_token_len, dtype=float)
                for x in range(bert_token_len):
                    bert_graph[x] = aspect_graph[text_bert__tokens_ids[x]]

                pad_dependency_graph = np.zeros((tokenizer.max_seq_len, tokenizer.max_seq_len)).astype('float32')
                pad_dependency_graph[1:bert_token_len + 1, 1:bert_token_len + 1] = bert_dependency_graph
                pad_graph = np.zeros(tokenizer.max_seq_len).astype('float32')
                pad_graph[1:bert_token_len + 1] = bert_graph

                data = {
                    'concat_bert_indices': concat_bert_indices,
                    'concat_segments_indices': concat_segments_indices,
                    'text_bert_indices': text_bert_indices,
                    'aspect_bert_indices': aspect_bert_indices,
                    'dependency_graph': pad_dependency_graph,
                    'polarity': polarity,
                    'cl_polarity': cl_polarity,
                    'aspect_mask': aspect_mask,
                    'new_graph': pad_graph,
                }

                all_data.append(data)
        else:
            for i in range(0, len(lines), 3):
                text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
                aspect = lines[i + 1].lower().strip()
                polarity = lines[i + 2].strip()
                text = text_left + ' ' + aspect + ' ' + text_right

                text_indices = tokenizer.text_to_sequence(text_left + " " + aspect + " " + text_right)
                aspect_indices = tokenizer.text_to_sequence(aspect)
                aspect_len = np.sum(aspect_indices != 0)
                text_len = np.sum(text_indices != 0)
                polarity = int(polarity) + 1

                concat_bert_indices = tokenizer.text_to_sequence(
                    '[CLS] ' + text_left + " " + aspect + " " + text_right + ' [SEP] ' + aspect + " [SEP]")
                concat_segments_indices = [0] * (text_len + 2) + [1] * (aspect_len + 1)
                concat_segments_indices = pad_and_truncate(concat_segments_indices, tokenizer.max_seq_len)
                text_bert_indices = tokenizer.text_to_sequence(
                    "[CLS] " + text_left + " " + aspect + " " + text_right + " [SEP]")
                aspect_bert_indices = tokenizer.text_to_sequence("[CLS] " + aspect + " [SEP]")

                aspect_mask = ABSADataset.__get_aspect_mask__(aspect, concat_bert_indices, tokenizer)
                aspect_graph = new_graph[i]['graph']

                text_bert_tokens = []
                text_bert__tokens_ids = []
                for token_id, token in enumerate(text.split()):
                    for bert_token in tokenizer.text_to_sequence4bert(token):
                        text_bert_tokens.append(bert_token)
                        text_bert__tokens_ids.append(token_id)

                bert_token_len = len(tokenizer.text_to_sequence4bert(text))
                dependency_graph = idx2graph[i]
                bert_dependency_graph = np.zeros((bert_token_len, bert_token_len), dtype=float)
                for x in range(bert_token_len):
                    for y in range(bert_token_len):
                        bert_dependency_graph[x][y] = dependency_graph[text_bert__tokens_ids[x]][
                            text_bert__tokens_ids[y]]

                bert_graph = np.zeros(bert_token_len, dtype=float)
                for x in range(bert_token_len):
                    bert_graph[x] = aspect_graph[text_bert__tokens_ids[x]]

                pad_dependency_graph = np.zeros((tokenizer.max_seq_len, tokenizer.max_seq_len)).astype('float32')
                pad_dependency_graph[1:bert_token_len + 1, 1:bert_token_len + 1] = bert_dependency_graph
                pad_graph = np.zeros(tokenizer.max_seq_len).astype('float32')
                pad_graph[1:bert_token_len + 1] = bert_graph

                data = {
                    'text': text,
                    'aspect': aspect,
                    'concat_bert_indices': concat_bert_indices,
                    'concat_segments_indices': concat_segments_indices,
                    'text_bert_indices': text_bert_indices,
                    'aspect_bert_indices': aspect_bert_indices,
                    'dependency_graph': pad_dependency_graph,
                    'polarity': polarity,
                    'aspect_mask': aspect_mask,
                    'new_graph': pad_graph,
                }

                all_data.append(data)


        self.data = all_data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    @staticmethod
    def __get_aspect_mask__(aspect, text_bert_indices, tokenizer):
        aspect = tokenizer.tokenizer.encode(aspect)
        aspect.pop(0)
        aspect.pop(-1)
        aspect_len = len(aspect)
        aspect_mask = torch.zeros_like(torch.LongTensor(text_bert_indices))
        for x in range(len(text_bert_indices)):
            try:
                if aspect == text_bert_indices[x:x + aspect_len]:
                    aspect_mask[x:x + aspect_len] = 1
                    break
            except:
                if (aspect == text_bert_indices[x:x + aspect_len]).all():
                    aspect_mask[x:x + aspect_len] = 1
                    break
        return aspect_mask.squeeze(dim=0)


