import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout, alpha, concat=True):
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

    def forward(self, x, adj):
        """
        x:[N, input_dim]
        adj:[N, N]
        W:[input_dim, output_dim]
        """
        h = torch.mm(x, self.W)  # h = xW = [N, output_dim]
        N = h.size()[0]  # N = 数量，class<int>

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1,
                                                                                          2 * self.output_dim)  # 下一代码块看看这部分代码作用
        e = self.relu(torch.matmul(a_input, self.a).squeeze(2))  # 原文eq.3 LeakyReLU部分

        zero_vec = -1e12 * torch.ones_like(e)  # 无穷小
        attention = torch.where(adj > 0, e, zero_vec)  # 通过邻接矩阵adj将不相邻的节点置无穷小

        attention = F.softmax(attention, dim=1)  # eq.2 归一化注意力系数
        attention = F.dropout(attention, self.dropout)
        h_prime = torch.matmul(attention, h)  # eq.5

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.input_dim) + ' -> ' + str(self.output_dim) + ')'

