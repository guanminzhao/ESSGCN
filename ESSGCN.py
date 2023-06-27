import torch
import torch.nn.functional as F
import torch.nn as nn
from layers.graph_convolution import GraphConvolution
from layers.GAT import MyGraphAttentionLayer

class ESSGCN(nn.Module):
    def __init__(self, bert, opt):
        super(ESSGCN, self).__init__()
        self.opt = opt
        self.bert = bert

        self.dropout = nn.Dropout(opt.dropout)
        self.gat = MyGraphAttentionLayer(opt.bert_dim, opt.bert_dim, opt.gat_dropout, opt.dropout)
        self.gcn1 = GraphConvolution(opt.bert_dim, opt.bert_dim)
        self.gcn2 = GraphConvolution(opt.bert_dim, opt.bert_dim)
        self.gcn3 = GraphConvolution(opt.bert_dim, opt.bert_dim)
        self.gcn4 = GraphConvolution(opt.bert_dim, opt.bert_dim)
        self.ffn = nn.Sequential(
            nn.Linear(opt.bert_dim, opt.bert_dim),
            nn.ReLU(),
            nn.Dropout(opt.ffn_dropout),  # pool = max 时0.3为最好，pool = mean 时0.5最好
            nn.Linear(opt.bert_dim, opt.polarities_dim)
        )
        self.alpha = 0.7
        self.beta = 0.3
        self.W1 = nn.Parameter(torch.Tensor(opt.bert_dim, opt.bert_dim))
        opt.initializer(self.W1)
        self.W2 = nn.Parameter(torch.Tensor(opt.bert_dim, opt.bert_dim))
        opt.initializer(self.W2)
        self.W3 = nn.Parameter(torch.Tensor(opt.bert_dim, opt.bert_dim))
        opt.initializer(self.W3)
        self.W_L = nn.Parameter(torch.Tensor(opt.bert_dim, opt.bert_dim))
        opt.initializer(self.W_L)
        self.L0 = nn.Parameter(torch.Tensor(1,opt.bert_dim))
        opt.initializer(self.L0)
        self.L1 = nn.Parameter(torch.Tensor(1, opt.bert_dim))
        opt.initializer(self.L1)


    def get_sem_graph(self, attentions, num, alpha):
        num = eval(num)
        attention = 0
        for i in num:
            attention += attentions[i].mean(dim=1)
        attention = attention / len(num)
        sem_graph = torch.where(attention > alpha, 1.0, 0.0)
        return sem_graph, attention

    def BiAffine(self, H1, W, H2):
        H_out = torch.matmul(H1, W)
        H_out = torch.matmul(H_out, H2.permute(0, 2, 1))
        H_out = torch.matmul(F.softmax(H_out, dim=-1), H1)
        return H_out

    def get_penal(self, Sem_graph):
        Sem_graph_T = Sem_graph.transpose(1, 2)
        identity = torch.eye(Sem_graph.size(1)).to(self.opt.device)
        identity = identity.unsqueeze(0).expand(Sem_graph.size(0), Sem_graph.size(1), Sem_graph.size(1))
        ortho = Sem_graph @ Sem_graph_T

        for i in range(ortho.size(0)):
            ortho[i] -= torch.diag(torch.diag(ortho[i]))
            ortho[i] += torch.eye(ortho[i].size(0)).to(self.opt.device)
        penal = (torch.norm(ortho - identity) / Sem_graph.size(0)).to(self.opt.device)
        return penal

    def forward(self, inputs):
        text_bert_indices, bert_segments_ids, dependency_graph = inputs[0], inputs[1], inputs[2]
        graph, aspect_mask = inputs[3], inputs[4]
        output = self.bert(input_ids=text_bert_indices, token_type_ids=bert_segments_ids, output_attentions=True)

        _, y = self.gat(output[0], aspect_mask, graph)
        h_syn = self.gcn1(output[0], dependency_graph)
        h_syn = self.gcn2(h_syn, dependency_graph)
        mask_index = aspect_mask.unsqueeze(dim=-1).repeat(1, 1, self.opt.bert_dim)

        Sem_graph, origin_graph = self.get_sem_graph(output['attentions'], self.opt.num, self.opt.k)
        h_sem = self.gcn3(output[0], Sem_graph)
        h_sem = self.gcn4(h_sem, Sem_graph)

        H_sem = self.alpha * output[0] + self.beta * h_sem
        H_syn = self.BiAffine(h_syn, self.W3, y)
        H_sem_ = self.BiAffine(H_sem, self.W1, H_syn)
        H_syn_ = self.BiAffine(H_syn, self.W2, H_sem)
        h_sem_a = (self.dropout(H_sem_) * mask_index).sum(dim=1) / mask_index.sum(dim=1)
        h_syn_a = (self.dropout(H_syn_) * mask_index).sum(dim=1) / mask_index.sum(dim=1)
        alpha = (0.5 + 0.25 * torch.cosine_similarity(torch.matmul(h_syn_a, self.W_L), self.L0) -\
                0.25 * torch.cosine_similarity(torch.matmul(h_syn_a, self.W_L), self.L1)).unsqueeze(dim=1)
        alpha = alpha.repeat(1, self.opt.bert_dim)
        logits = self.ffn(((1-alpha) * h_sem_a + alpha * h_syn_a))
        penal = self.get_penal(origin_graph)
        return logits, h_sem_a, h_syn_a, penal

