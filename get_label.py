import spacy
import torch
import pickle
from tqdm import tqdm
import argparse


def get_label(nlp1, nlp2, lines, fin):


    for i in tqdm(range(0, len(lines), 3)):
        text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
        aspect = lines[i + 1].lower().strip()


        text = text_left + ' ' + aspect + ' ' + text_right
        doc1 = nlp1(text)
        len1 = len(doc1)
        adj1 = torch.zeros([len1, len1], dtype=torch.int)
        for token in doc1:
            adj1[token.i][token.head.i] = 1
            adj1[token.head.i][token.i] = 1

        doc2 = nlp2(text)
        len2 = len(doc2)
        adj2 = torch.zeros([len2, len2], dtype=torch.int)
        for token in doc2:
            adj2[token.i][token.head.i] = 1
            adj2[token.head.i][token.i] = 1

        num = torch.abs(adj1 - adj2).sum()
        if num == 0:
            new_polarity = 0
        else:
            new_polarity = 1
        fin.write(lines[i])
        fin.write(lines[i + 1])
        fin.write(lines[i + 2])
        fin.write(str(new_polarity) + '\n')
    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_file', type=str)
    parser.add_argument('--target_path', type=str)
    opt = parser.parse_args()

    nlp1 = spacy.load('en_core_web_lg')
    nlp2 = spacy.load('en_core_web_md')
    fin = open(opt.source_file, 'r')
    lines = fin.readlines()
    fin.close()
    fin = open(opt.target_path, 'w', encoding='utf-8', newline='\n', errors='ignore')

    get_label(nlp1, nlp2, lines, fin)
