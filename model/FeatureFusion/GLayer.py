import torch
import torch.nn as nn
import torch.nn.functional as F


class gconv(nn.Module):
    def __init__(self):
        super(gconv, self).__init__()

    def forward(self, x, A):
        if len(x.shape) == 4:
            x = torch.einsum('bcnl, nw->bcwl', (x, A))

        elif len(x.shape) == 3:
            x = torch.einsum('ncl,cw->nwl', (x, A))
        return x.contiguous()


class prop(nn.Module):
    def __init__(self, c_in, c_out, gdep, dropout, alpha):
        super(prop, self).__init__()
        self.g_conv = gconv()
        self.mlp = nn.Conv2d(c_in, c_out, 1, 1, bias=True)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha

    def forward(self, x, adj):
        adj = adj + torch.eye(adj.size(0)).to(x.device)
        d = adj.sum(1)
        h = x
        dv = d
        a = adj / dv.view(-1, 1)
        for i in range(self.gdep):
            h = self.alpha * x + (1 - self.alpha) * self.g_conv(h, a)
        ho = self.mlp(h)
        return ho


class mixprop(nn.Module):
    def __init__(self, c_in, c_out, gdep, dropout, alpha):
        super(mixprop, self).__init__()
        self.g_conv = gconv()
        self.gdep = gdep
        self.mlp = nn.Conv2d((gdep + 1) * c_in, c_out, 1, 1)
        self.dropout = dropout
        self.alpha = alpha

    def forward(self, x, adj):
        adj = adj + torch.eye(adj.size(0)).to(x.device)
        d = adj.sum(1)
        h = x
        out = [x]
        a = adj / d.view(-1, 1)
        for i in range(self.gdep):
            h = (1 - self.alpha) * x + self.alpha * self.g_conv(h, a)
            out.append(h)

        ho = torch.cat(out, dim=1)
        ho = self.mlp(ho)

        return ho


class mixgcn(nn.Module):
    def __init__(self, c_in, c_out, gdep, dropout, alpha):
        super(mixgcn, self).__init__()
        self.g_conv = gconv()
        self.gdep = gdep
        self.mlp = nn.Conv2d((gdep + 1) * c_in, c_out, 1, 1)
        self.dropout = dropout
        self.alpha = alpha

    def forward(self, x, adj):
        adj1 = torch.where(adj != 0, torch.tensor(1.0, device=adj.device), adj) + torch.eye(adj.size(0)).to(x.device)
        adj = adj + torch.eye(adj.size(0)).to(x.device)
        d = adj1.sum(1)
        h = x
        out = [x]
        a = adj / d.view(-1, 1)
        for i in range(self.gdep):
            h = (1 - self.alpha) * x + self.alpha * self.g_conv(h, a)
            out.append(h)

        ho = torch.cat(out, dim=1)
        ho = self.mlp(ho)

        # return torch.relu(ho)
        return ho


class graph_constructor(nn.Module):
    def __init__(self, nnodes, dim, f_num, k, alpha, device):
        super(graph_constructor, self).__init__()
        self.nnodes = nnodes
        self.layers = f_num

        self.emb1 = nn.Embedding(nnodes, dim)
        self.emb2 = nn.Embedding(nnodes, dim)

        self.lin1 = nn.ModuleList()
        self.lin2 = nn.ModuleList()
        for i in range(f_num):
            self.lin1.append(nn.Linear(dim, dim))
            self.lin2.append(nn.Linear(dim, dim))

        self.device = device
        self.k = k
        self.dim = dim
        self.alpha = alpha

    def forward(self, idx, scale_set):

        nodevec1 = self.emb1(idx)
        nodevec2 = self.emb2(idx)

        adj_set = []

        for i in range(self.layers):
            nodevec1 = torch.tanh(self.alpha * self.lin1[i](nodevec1 * scale_set[i]))
            nodevec2 = torch.tanh(self.alpha * self.lin2[i](nodevec2 * scale_set[i]))
            a = torch.mm(nodevec1, nodevec2.transpose(1, 0)) - torch.mm(nodevec2, nodevec1.transpose(1, 0))
            adj0 = F.relu(torch.tanh(self.alpha * a))

            mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)
            val, ind = adj0.topk(self.k, 1)
            mask.scatter_(1, ind, val.fill_(1))
            # print(mask)
            adj = adj0 * mask
            adj_set.append(adj)

        return adj_set


class ScoreGraph(nn.Module):
    def __init__(self, nodes, dim, f_num, k, alpha, device):
        super(ScoreGraph, self).__init__()
        self.nodes = nodes
        self.f_num = f_num
        self.emb1 = nn.Embedding(nodes, dim)
        self.emb2 = nn.Embedding(nodes, dim)

        self.lin1 = nn.ModuleList([nn.Linear(dim, dim) for _ in range(f_num)])
        self.lin2 = nn.ModuleList([nn.Linear(dim, dim) for _ in range(f_num)])

        self.device = device
        self.alpha = alpha
        self.k = k
        self.dim = dim

    def forward(self, idx):
        nodevec1 = self.emb1(idx)
        nodevec2 = self.emb2(idx)
        adj_set = []

        for i in range(self.f_num):
            vec1 = torch.tanh(self.alpha * self.lin1[i](nodevec1))
            vec2 = torch.tanh(self.alpha * self.lin2[i](nodevec2))
            a = torch.mm(vec1, vec2.transpose(1, 0)) - torch.mm(vec2, vec1.transpose(1, 0))
            adj0 = F.relu(torch.tanh(self.alpha * a))

            mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)
            val, ind = adj0.topk(self.k, 1)
            mask.scatter_(1, ind, val.fill_(1))
            # mask.scatter_(1, ind, val)
            adj = adj0 * mask
            adj_set.append(adj)

        return adj_set
