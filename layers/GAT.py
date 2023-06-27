import torch
import torch.nn as nn
import torch.nn.functional as F


class MyGraphAttentionLayer(nn.Module):
    def __init__(self, input_dim, output_dim, alpha, dropout, concat=True):
        super(MyGraphAttentionLayer, self).__init__()
        self.input_dim = input_dim  # hidden_embed
        self.output_dim = output_dim  # 自己定义
        self.dropout = dropout  # 自己决定
        self.alpha = alpha  # 自己决定
        self.concat = concat
        self.W = nn.Parameter(torch.zeros(size=(input_dim, output_dim)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)  # 初始化W参数，使其服从均匀分布
        self.a = nn.Parameter(torch.zeros(size=(2 * output_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.relu = nn.LeakyReLU(self.alpha)

    def forward(self, input, mask_index, adj):


        '''
        input.shape = [seq_len, hidden_embed]
        mask_index.shape = [1, seq_len]
        adj.shape = [1, seq_len]
        '''
        h = torch.matmul(input, self.W)  # [bzs, seq_len, output_dim]
        batch_size = h.size()[0]
        seq_len = h.size()[1]  # seq_len
        if len(mask_index.size()) < 3:
            mask_index = mask_index.unsqueeze(dim=-1)
        mask_index = mask_index.repeat(1, 1, self.output_dim)
        h_star = h * mask_index
        h_star = h_star.sum(dim=1)/mask_index.sum(dim=1)  # [bzs, output_dim] 找出aspect对应的词向量，如果有多个词那么取平均
        e = torch.matmul(torch.cat((h_star.repeat(1, seq_len).view(seq_len * batch_size, -1), h.view(-1, self.output_dim)), dim=1), self.a)  # [bzs×seq_len,1]
        e = self.relu(e)  # [bzs×seq_len,1]
        e = e.view(batch_size, seq_len, -1).permute(0, 2, 1)  # [bzs, 1, seq_len]
        zero_vec = -9e15 * torch.ones_like(e)  # [bzs, 1, seq_len]
        if len(adj.size()) < 3:
            adj = adj.unsqueeze(dim=1)
        attention = torch.where(adj > 0, e, zero_vec)  # [bzs, 1, seq_len]
        attention = F.softmax(attention, dim=2)  # [bzs, 1, seq_len]
        attention = F.dropout(attention, self.dropout, training=self.training)  # [bzs, 1, seq_len]
        h_prime = torch.matmul(attention, h)  # [bzs, 1, output_dim]
        if self.concat:
            h_prime = F.elu(h_prime)
            output = h_prime.repeat(1, seq_len, 1) * mask_index + h * (1-mask_index)
            return h_prime, output  # 公式4
        else:
            output = h_prime.repeat(1, seq_len, 1) * mask_index + h * (1 - mask_index)
            return h_prime, output

class MyGAT(nn.Module):
    def __init__(self, input_dim, nheads, alpha, dropout, single=True):
        super(MyGAT, self).__init__()
        self.single = single
        self.dropout = dropout
        if input_dim % nheads != 0:
            raise ValueError('Error nheads: {}'.format(nheads))
        output_dim = int(input_dim/nheads)
        self.attentions = [MyGraphAttentionLayer(input_dim, output_dim, dropout=dropout, alpha=alpha, concat=True) for _
                           in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

    def forward(self, x, mask_index, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        if self.single:
            x = torch.cat([att(x, mask_index, adj)[0] for att in self.attentions], dim=2)
        else:
            x = torch.cat([att(x, mask_index, adj)[1] for att in self.attentions], dim=2)
        x = F.dropout(x, self.dropout, training=self.training)
        return x

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, input_dim, output_dim, alpha, dropout, concat=True):

        super(GraphAttentionLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat
        self.W = nn.Parameter(torch.zeros(size=(input_dim, output_dim)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)  # 初始化W参数，使其服从均匀分布
        self.a = nn.Parameter(torch.zeros(size=(2 * output_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.relu = nn.LeakyReLU(self.alpha)


    def forward(self, input, adj):
        h = torch.mm(input, self.W)#公式1：WH部分
        #W:(1433,8) input(2708,1433)->h:(2708,8) 直接通过torch.mm矩阵乘完成全连接运算
        #最后还有一次self-attention，由self.out_att(x, adj)传入得到的self.W:(64,8)，传入的input为(2707,64)
        N = h.size()[0]
        '''
        h.repeat(1, N):(2708,8*2708).view->(2708*2708,8),h.repeat(N, 1)->(2708*2708,8)
        torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1)->(2708*2708,16).view->2708,2708,2*8
        self.a:(16,1)偏移量b，torch.matmul
        结果如下：对于节点1，分别与节点1属性，节点2属性，。。。一直到节点2708属性合并：
        [节点1： [[节点1的8个属性，节点1的8个属性]
                [节点1的8个属性，节点2的8个属性]
                。。。。。。
                [节点1的8个属性，节点2708的8个属性]]     共2708个
        节点2：        
                [[节点2的8个属性，节点1的8个属性]
                [节点2的8个属性，节点2的8个属性]
                。。。。。。
                [节点2的8个属性，节点2708的8个属性]]     共2708个
        。。。
        节点2708：        
                [[节点2708的8个属性，节点1的8个属性]
                [节点2708的8个属性，节点2的8个属性]
                。。。。。。
                [节点2708的8个属性，节点2708的8个属性]]     共2708个  
        ]所以a_input是(2708,2708,16)
        这表明每个节点的8个属性(是经过mm运算将1403个属性集成为8个)与所有节点都进行了聚集。
        self-attention会将注意力分配到图中所有的节点上，这种做法显然会丢失结构信息。
        为了解决这一问题，本文使用了一种masked attention的方式——仅将注意力分配到节点的邻节点集上。
        也就是后面的根据邻接关系更新这个e：  attention = torch.where(adj > 0, e, zero_vec)
        '''
        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))#公式3里面的leakyrelu部分 即公式1的e
        #造了一个e:(2708,2708)维度,元素为全0的张量
        zero_vec = -9e15*torch.ones_like(e)
        #根据邻接矩阵adj，找出e中大于0的元素组成张量
        '''
        torch.where(condition,x,y)
        out = x,if condition is 1
            = y ,if condition is 0
        '''
        # 也就是根据邻接矩阵adj来组建attention(2708,2708)，若对应adj位置大于0，取e中对应位置元素，若小于0取为zero_vec中对应位置元素
        attention = torch.where(adj > 0, e, zero_vec)
        # 真正的公式3，单层全连接处理后只考虑邻接矩阵后的激活函数，就相当于一个考虑了邻居的权重系数
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)  # h_prime(2708,8)
        # 将这个权重系数attention与Wh相乘，即在原先节点属性运算时考虑了邻接节点的权重关系。
        # 相当于公式4的括号里面部分，只考虑了邻接节点
        if self.concat:
            return F.elu(h_prime)  # 公式4
        else:
            return h_prime







class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)


if __name__ == '__main__':
    model = MyGAT(input_dim=300, nheads=3, alpha=0.5, dropout=0.5)
    x = torch.randn(16, 15, 300)
    mask_index = torch.randint(0, 2, (16, 15))
    adj = torch.randint(0, 2, (16, 1, 15))
    model(x, mask_index, adj)