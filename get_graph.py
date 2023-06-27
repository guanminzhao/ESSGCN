import torch
from transformers import BertTokenizer, BertModel
import pickle
import copy
from tqdm import tqdm
import argparse

def process(bert, tokenizer, lines, f):
    dict = {}
    for i in tqdm(range(0, len(lines), 3)):
        text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
        aspect = lines[i + 1].lower().strip()
        polarity = lines[i + 2].strip()

        if len(aspect.split()) > 1:
            graph_list = []
            small_dict = {}
            for x in aspect.split():
                mask_aspect_text = text_left + ' ' + aspect.replace(x, '[MASK]') + ' ' + text_right
                dist = []
                with torch.no_grad():
                    for j in range(len(mask_aspect_text.split())):
                        list = mask_aspect_text.split()
                        list[j] = '[MASK]'
                        text = ' '.join(list)
                        input1 = tokenizer(mask_aspect_text, return_tensors='pt').to(opt.device)
                        input2 = tokenizer(text, return_tensors='pt').to(opt.device)
                        output1 = bert(input1['input_ids'])
                        output2 = bert(input2['input_ids'])
                        dist.append(torch.cdist(output1[1], output2[1], p=2))
                try:
                    a = torch.topk(torch.Tensor(dist), 3)
                except:
                    print(1)
                graph = torch.zeros_like(torch.Tensor(dist))
                for y in a[1]:
                    graph[y] = 1
                if not bool(small_dict):
                    small_dict['graph'] = graph
                    mask_index = torch.zeros_like(torch.Tensor(dist))
                    index = mask_aspect_text.split().index('[MASK]')
                    mask_index[index] = 1
                    small_dict['mask_index'] = mask_index
                else:
                    small_dict['graph'] = small_dict['graph'] + graph
                    small_dict['graph'] = torch.where(small_dict['graph'] >= 1, 1, 0)
                    index = mask_aspect_text.split().index('[MASK]')
                    small_dict['mask_index'][index] = 1
                    small_dict['graph'] = torch.where(small_dict['graph'] - small_dict['mask_index'] <= 0, 0, 1)
            dict[i] = small_dict
        else:
            small_dict = {}
            mask_aspect_text = text_left + ' ' + '[MASK]' + ' ' + text_right
            graph_list = []
            dist = []
            with torch.no_grad():
                for j in range(len(mask_aspect_text.split())):
                    list = mask_aspect_text.split()
                    list[j] = '[MASK]'
                    text = ' '.join(list)
                    input1 = tokenizer(mask_aspect_text, return_tensors='pt').to(opt.device)
                    input2 = tokenizer(text, return_tensors='pt').to(opt.device)
                    output1 = bert(input1['input_ids'])
                    output2 = bert(input2['input_ids'])
                    dist.append(torch.cdist(output1[1], output2[1], p=2))
            if(len(dist) >= 3):
                a = torch.topk(torch.Tensor(dist), 3)
                graph = torch.zeros_like(torch.Tensor(dist))
                for y in a[1]:
                    graph[y] = 1
                small_dict['graph'] = graph
                mask_index = torch.zeros_like(torch.Tensor(dist))
                index = mask_aspect_text.split().index('[MASK]')
                mask_index[index] = 1
                small_dict['mask_index'] = mask_index
                dict[i] = small_dict
            else:
                graph = torch.ones_like(torch.Tensor(dist))
                small_dict['graph'] = graph
                mask_index = torch.zeros_like(torch.Tensor(dist))
                index = mask_aspect_text.split().index('[MASK]')
                mask_index[index] = 1
                small_dict['mask_index'] = mask_index
                dict[i] = small_dict
    pickle.dump(dict, f)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default=None, type=str, help='e.g. cuda:0')
    parser.add_argument('--source_file', type=str)
    parser.add_argument('--target_path', type=str)
    opt = parser.parse_args()
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if opt.device is None else torch.device(opt.device)
    Bert_Path = './bert'
    bert = BertModel.from_pretrained(Bert_Path).to(opt.device)
    tokenizer = BertTokenizer.from_pretrained(Bert_Path)

    fin = open(opt.source_file, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines = fin.readlines()
    fin.close()

    fin = open(opt.target_path, 'wb')
    process(bert, tokenizer, lines, fin)
    fin.close()

