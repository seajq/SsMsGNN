import os, sys, torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from FeatureFusion.GLayer import graph_constructor, mixprop, mixgcn, ScoreGraph
from FeatureFusion.FPN import MultiScaleConv, FPNFusion
from FeatureFusion.MSFF import MSFF, AvgChannelFusion, MSFFusion


class CrossAE(nn.Module):

    def __init__(self, Encoder, Decoder, Classifier):
        super().__init__()

        self.encoder = Encoder
        self.decoder = Decoder
        self.classifier = Classifier

    def forward(self, input, pretrain=True):
        h = self.encoder(input)

        if pretrain:
            return self.decoder(h)
        else:
            return self.classifier(h)


class FreezeAE(nn.Module):

    def __init__(self, Encoder, Decoder, Classifier):
        super().__init__()

        self.encoder = Encoder
        self.decoder = Decoder
        self.classifier = Classifier

    def forward(self, input, pretrain=True):
        z = self.encoder(input)

        if pretrain:
            return self.decoder(z)
        else:
            return self.classifier(z)


class ClassificationHead(nn.Module):
    def __init__(self, in_features, hidden_features, num_classes):
        super(ClassificationHead, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_features, num_classes)

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class FPN(nn.Module):
    def __init__(self, kernel_set=None, in_channels=1, scale_channels=16, droupout=0.3, seq_length=120):
        super().__init__()
        if kernel_set is None:
            kernel_set = [7, 6, 3, 2]
        self.kernel_set = kernel_set
        self.dropout = droupout
        self.scale0 = nn.Conv2d(in_channels=in_channels, out_channels=scale_channels, kernel_size=(1, seq_length), bias=True)
        self.MultiScaleConv = MultiScaleConv(in_channels, scale_channels, seq_length, self.kernel_set)

    def forward(self, input):
        input = input.transpose(-2, -1).unsqueeze(dim=1) if input.dim() == 3 else input
        scale0 = F.dropout(self.scale0(input), p=self.dropout, training=self.training)
        scale = self.MultiScaleConv(input)

        return [scale0] + scale


class FPNGNN(nn.Module):
    def __init__(self, gcn_depth, num_nodes, device, kernel_set=[14, 7, 6, 3], dropout=0.3, subgraph_size=20, node_dim=40,
                 conv_channels=32, end_channels=128, out_channels=16,
                 seq_length=120, in_channels=1, propalpha=0.05, alpha=3, class_num=None):
        super().__init__()

        self.num_nodes = num_nodes
        self.device = device

        self.dropout = dropout
        self.gate_convs = nn.ModuleList()
        self.scale_convs = nn.ModuleList()

        self.gnet1 = nn.ModuleList()
        self.gnet2 = nn.ModuleList()
        self.fpn = FPN(kernel_set, in_channels, conv_channels, dropout, seq_length)
        self.idx = torch.arange(self.num_nodes).to(device)
        self.gc = graph_constructor(num_nodes, node_dim, len(kernel_set), subgraph_size, alpha, device)

        self.kernel_set = kernel_set
        self.idx = torch.arange(self.num_nodes).to(device)

        length_set = []
        length_set.append(seq_length - self.kernel_set[0] + 1)
        for i in range(1, len(kernel_set)):
            length_set.append(int((length_set[i - 1] - self.kernel_set[i]) / 2))

        for i in range(len(kernel_set)):
            self.gnet1.append(mixprop(conv_channels, conv_channels, gcn_depth, dropout, propalpha))
            self.gnet2.append(mixprop(conv_channels, conv_channels, gcn_depth, dropout, propalpha))

            self.scale_convs.append(nn.Conv2d(in_channels=conv_channels,
                                              out_channels=conv_channels,
                                              kernel_size=(1, length_set[i])))

        self.fpngate = FPNFusion(conv_channels, len(kernel_set) + 1)

        self.end_conv = nn.Sequential(nn.Conv2d(in_channels=conv_channels, out_channels=end_channels, kernel_size=(1, 1), bias=True),
                                      nn.ReLU(),
                                      nn.Conv2d(in_channels=end_channels, out_channels=out_channels, kernel_size=(1, 1), bias=True), )

        self.endlin = nn.Linear(num_nodes, class_num)

    def forward(self, input):
        scale = self.fpn(input)
        scale0 = scale[0]
        scale = scale[1:]

        adj_matrix = self.gc(self.idx, [1, 0.8, 0.6, 0.5])
        out = [scale0]

        for i in range(len(self.kernel_set)):
            output = self.gnet1[i](scale[i], adj_matrix[i]) + self.gnet2[i](scale[i], adj_matrix[i].transpose(1, 0))
            out.append(self.scale_convs[i](output))

        fpngate_out = self.fpngate(out)
        pred = self.end_conv(F.relu(fpngate_out)).squeeze()
        logits = self.endlin(pred)

        return logits


class FPNEncoder(nn.Module):
    def __init__(self, gcn_depth, num_nodes, device, kernel_set=[14, 7, 6, 3], dropout=0.3, subgraph_size=20, node_dim=40,
                 conv_channels=32, end_channels=128, out_channels=16,
                 seq_length=120, in_channels=1, propalpha=0.05, alpha=3, class_num=None):
        super().__init__()
        self.device = device
        self.num_nodes = num_nodes
        self.dropout = dropout

        self.fpn = FPN(kernel_set, in_channels, conv_channels, dropout, seq_length)
        self.gnet1 = nn.ModuleList()
        self.gnet2 = nn.ModuleList()
        self.idx = torch.arange(self.num_nodes).to(device)
        self.gc = graph_constructor(num_nodes, node_dim, len(kernel_set), subgraph_size, alpha, device)
        self.kernel_set = kernel_set

        for i in range(len(kernel_set)):
            self.gnet1.append(mixprop(conv_channels, conv_channels, gcn_depth, dropout, propalpha))
            self.gnet2.append(mixprop(conv_channels, conv_channels, gcn_depth, dropout, propalpha))

    def forward(self, input):
        temp = self.fpn(input)
        scale0 = temp[0]
        scale = temp[1:]

        adj_matrix = self.gc(self.idx, [1, 0.8, 0.6, 0.5])
        out = [scale0]

        for i in range(len(self.kernel_set)):
            output = self.gnet1[i](scale[i], adj_matrix[i]) + self.gnet2[i](scale[i], adj_matrix[i].transpose(1, 0))
            out.append(output)

        return out


class FPNDecoder(nn.Module):
    def __init__(self, gcn_depth, num_nodes, device, kernel_set=[14, 7, 6, 3], dropout=0.3, subgraph_size=20, node_dim=40,
                 conv_channels=32, end_channels=128, out_channels=16,
                 seq_length=120, in_channels=1, propalpha=0.05, alpha=3, class_num=None):
        super().__init__()

        self.device = device
        self.num_nodes = num_nodes
        self.dropout = dropout

        self.gnet1 = nn.ModuleList()
        self.gnet2 = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.scale_convs = nn.ModuleList()

        self.idx = torch.arange(self.num_nodes).to(device)
        self.gc = graph_constructor(num_nodes, node_dim, len(kernel_set), subgraph_size, alpha, device)

        self.kernel_set = kernel_set

        length_set = []
        length_set.append(seq_length - self.kernel_set[0] + 1)
        for i in range(1, len(kernel_set)):
            length_set.append(int((length_set[i - 1] - self.kernel_set[i]) / 2))

        for i in range(len(kernel_set)):
            self.gnet1.append(mixprop(conv_channels, conv_channels, gcn_depth, dropout, propalpha))
            self.gnet2.append(mixprop(conv_channels, conv_channels, gcn_depth, dropout, propalpha))

            self.scale_convs.append(nn.Conv2d(in_channels=conv_channels,
                                              out_channels=conv_channels,
                                              kernel_size=(1, length_set[i])))

        self.fpngate = FPNFusion(conv_channels, len(kernel_set) + 1)

        self.end_conv = nn.Sequential(nn.Conv2d(in_channels=conv_channels, out_channels=end_channels, kernel_size=(1, 1), bias=True),
                                      nn.ReLU(),
                                      nn.Conv2d(in_channels=end_channels, out_channels=out_channels, kernel_size=(1, 1), bias=True), )

    def forward(self, input):

        scale0 = input[0]
        scale = input[1:]

        adj_matrix = self.gc(self.idx, [1, 0.8, 0.6, 0.5])
        out = [scale0]

        for i in range(len(self.kernel_set)):
            output = self.gnet1[i](scale[i], adj_matrix[i]) + self.gnet2[i](scale[i], adj_matrix[i].transpose(1, 0))
            out.append(self.scale_convs[i](output))

        fpngate_out = self.fpngate(out)
        pred = self.end_conv(F.relu(fpngate_out)).squeeze()

        return pred


class FPNClassifier(nn.Module):
    def __init__(self, gcn_depth, num_nodes, device, kernel_set=[14, 7, 6, 3], dropout=0.3, subgraph_size=20, node_dim=40,
                 conv_channels=32, end_channels=128, out_channels=16,
                 seq_length=120, in_channels=1, propalpha=0.05, alpha=3, class_num=None):
        super().__init__()

        self.num_nodes = num_nodes
        self.device = device
        self.dropout = dropout
        self.kernel_set = kernel_set
        self.gate_convs = nn.ModuleList()
        self.scale_convs = nn.ModuleList()
        self.gnet1 = nn.ModuleList()
        self.gnet2 = nn.ModuleList()
        self.gc = graph_constructor(num_nodes, node_dim, len(kernel_set), subgraph_size, alpha, device)
        self.idx = torch.arange(self.num_nodes).to(device)

        length_set = []
        length_set.append(seq_length - kernel_set[0] + 1)
        for i in range(1, len(kernel_set)):
            length_set.append(int((length_set[i - 1] - kernel_set[i]) / 2))

        for i in range(len(kernel_set)):
            self.gnet1.append(mixprop(conv_channels, conv_channels, gcn_depth, dropout, propalpha))
            self.gnet2.append(mixprop(conv_channels, conv_channels, gcn_depth, dropout, propalpha))

            self.scale_convs.append(nn.Conv2d(in_channels=conv_channels,
                                              out_channels=conv_channels,
                                              kernel_size=(1, length_set[i])))

        self.fpngate = FPNFusion(conv_channels, len(kernel_set) + 1)

        self.end_conv = nn.Sequential(nn.Conv2d(in_channels=conv_channels, out_channels=end_channels, kernel_size=(1, 1), bias=True),
                                      nn.ReLU(),
                                      nn.Conv2d(in_channels=end_channels, out_channels=out_channels, kernel_size=(1, 1), bias=True), )

        self.clf = nn.Linear(num_nodes, class_num)

    def forward(self, input):
        scale0 = input[0]
        scale = input[1:]

        adj_matrix = self.gc(self.idx, [1, 0.8, 0.6, 0.5])
        out = [scale0]

        for i in range(len(self.kernel_set)):
            output = self.gnet1[i](scale[i], adj_matrix[i]) + self.gnet2[i](scale[i], adj_matrix[i].transpose(1, 0))
            out.append(self.scale_convs[i](output))

        fpngate_out = self.fpngate(out)
        pred = self.end_conv(F.relu(fpngate_out)).squeeze()
        logits = self.clf(pred)

        return logits


class MSFGNN(nn.Module):
    def __init__(self, gcn_depth, num_nodes, device, kernel_set=[3, 7, 12, 24], dropout=0.3, subgraph_size=20, node_dim=40,
                 conv_channels=32, end_channels=128, out_channels=16,
                 seq_length=120, in_channels=1, propalpha=0.05, alpha=3, class_num=None):
        super().__init__()

        self.num_nodes = num_nodes
        self.device = device

        self.scale_convs = nn.ModuleList()
        self.gnet1 = nn.ModuleList()
        self.gnet2 = nn.ModuleList()

        self.MSFF = MSFF(in_channels, conv_channels, kernel_set)
        self.gc = ScoreGraph(num_nodes, node_dim, len(kernel_set), subgraph_size, alpha, device)

        self.kernel_set = kernel_set
        self.scale_num = len(kernel_set)
        length_set = [int((seq_length - self.kernel_set[i]) / 2) + 1 for i in range(len(kernel_set))]

        for i in range(len(kernel_set)):
            self.gnet1.append(mixprop(conv_channels, conv_channels, gcn_depth, dropout, propalpha))
            self.gnet2.append(mixprop(conv_channels, conv_channels, gcn_depth, dropout, propalpha))
            self.scale_convs.append(nn.Conv2d(in_channels=conv_channels, out_channels=conv_channels, kernel_size=(1, length_set[i])))

        self.scale_num = len(kernel_set)
        self.msffgate = MSFFusion(conv_channels, len(kernel_set))

        self.end_conv = nn.Sequential(nn.Conv2d(in_channels=conv_channels * len(kernel_set), out_channels=end_channels, kernel_size=(1, 1)),
                                      nn.Conv2d(in_channels=end_channels, out_channels=out_channels, kernel_size=(1, 1)), )
        self.clf = nn.Linear(num_nodes, class_num)

    def forward(self, input):

        scale = self.MSFF(input)
        adj_matrix = self.gc(torch.arange(self.num_nodes).to(self.device))
        out = []

        for i in range(len(self.kernel_set)):
            output = self.gnet1[i](scale[i], adj_matrix[i]) + self.gnet2[i](scale[i], adj_matrix[i].transpose(1, 0))
            out.append(self.scale_convs[i](output))

        fusion_out = self.msffgate(out)
        pred = self.end_conv(fusion_out).squeeze()
        logits = self.clf(pred)

        return logits


class MSFEncoder(nn.Module):
    def __init__(self, gcn_depth, num_nodes, device, kernel_set=[3, 7, 12, 24], dropout=0.3, subgraph_size=20, node_dim=40,
                 conv_channels=32, end_channels=128, out_channels=16,
                 seq_length=120, in_channels=1, propalpha=0.05, alpha=3, class_num=None):
        super().__init__()

        self.num_nodes = num_nodes
        self.device = device

        self.scale_convs = nn.ModuleList()
        self.gnet1 = nn.ModuleList()
        self.gnet2 = nn.ModuleList()

        self.MSFF = MSFF(in_channels, conv_channels, kernel_set)
        self.gc = ScoreGraph(num_nodes, node_dim, len(kernel_set), subgraph_size, alpha, device)

        self.kernel_set = kernel_set
        self.scale_num = len(kernel_set)

        for i in range(len(kernel_set)):
            self.gnet1.append(mixprop(conv_channels, conv_channels, gcn_depth, dropout, propalpha))
            self.gnet2.append(mixprop(conv_channels, conv_channels, gcn_depth, dropout, propalpha))

    def forward(self, input):

        scale = self.MSFF(input)
        adj_matrix = self.gc(torch.arange(self.num_nodes).to(input.device))
        out = []

        for i in range(len(self.kernel_set)):
            output = self.gnet1[i](scale[i], adj_matrix[i]) + self.gnet2[i](scale[i], adj_matrix[i].transpose(1, 0))
            out.append(output)
        return out


class MSFDecoder(nn.Module):
    def __init__(self, gcn_depth, num_nodes, device, kernel_set=[3, 7, 12, 24], dropout=0.3, subgraph_size=20, node_dim=40,
                 conv_channels=32, end_channels=128, out_channels=16,
                 seq_length=120, in_channels=1, propalpha=0.05, alpha=3, class_num=None):
        super().__init__()

        self.num_nodes = num_nodes
        self.device = device

        self.scale_convs = nn.ModuleList()
        self.gnet1 = nn.ModuleList()
        self.gnet2 = nn.ModuleList()

        self.gc = ScoreGraph(num_nodes, node_dim, len(kernel_set), subgraph_size, alpha, device)

        self.kernel_set = kernel_set
        self.scale_num = len(kernel_set)
        self.msffgate = MSFFusion(conv_channels, len(kernel_set))

        length_set = [int((seq_length - self.kernel_set[i]) / 2) + 1 for i in range(len(kernel_set))]

        for i in range(len(kernel_set)):
            self.gnet1.append(mixprop(conv_channels, conv_channels, gcn_depth, dropout, propalpha))
            self.gnet2.append(mixprop(conv_channels, conv_channels, gcn_depth, dropout, propalpha))

            self.scale_convs.append(nn.Conv2d(in_channels=conv_channels, out_channels=conv_channels, kernel_size=(1, length_set[i])))

        self.end_conv = nn.Sequential(nn.Conv2d(in_channels=conv_channels * len(kernel_set), out_channels=end_channels, kernel_size=(1, 1)),
                                      nn.ReLU(),
                                      nn.Conv2d(in_channels=end_channels, out_channels=out_channels, kernel_size=(1, 1)), )

    def forward(self, input):

        scale = input
        adj_matrix = self.gc(torch.arange(self.num_nodes).to(self.device))
        out = []

        for i in range(len(self.kernel_set)):
            output = self.gnet1[i](scale[i], adj_matrix[i]) + self.gnet2[i](scale[i], adj_matrix[i].transpose(1, 0))
            out.append(self.scale_convs[i](output))

        fusion_out = self.msffgate(out)
        pred = self.end_conv(fusion_out).squeeze()

        return pred


class MSFClassifier(nn.Module):
    def __init__(self, gcn_depth, num_nodes, device, kernel_set=[3, 7, 12, 24], dropout=0.3, subgraph_size=20, node_dim=40,
                 conv_channels=32, end_channels=128, out_channels=16,
                 seq_length=120, in_channels=1, propalpha=0.05, alpha=3, class_num=None):
        super().__init__()

        self.device = device
        self.scale_convs = nn.ModuleList()
        self.gnet1 = nn.ModuleList()
        self.gnet2 = nn.ModuleList()
        self.kernel_set = kernel_set
        self.num_nodes = num_nodes
        self.msffgate = MSFFusion(conv_channels, len(kernel_set))
        self.gc = ScoreGraph(num_nodes, node_dim, len(kernel_set), subgraph_size, alpha, device)

        length_set = [int((seq_length - self.kernel_set[i]) / 2) + 1 for i in range(len(kernel_set))]

        for i in range(len(kernel_set)):
            self.gnet1.append(mixprop(conv_channels, conv_channels, gcn_depth, dropout, propalpha))
            self.gnet2.append(mixprop(conv_channels, conv_channels, gcn_depth, dropout, propalpha))
            self.scale_convs.append(nn.Conv2d(in_channels=conv_channels, out_channels=conv_channels, kernel_size=(1, length_set[i])))

        self.end_conv = nn.Sequential(nn.Conv2d(in_channels=conv_channels * len(kernel_set), out_channels=end_channels, kernel_size=(1, 1)),
                                      nn.ReLU(),
                                      nn.Conv2d(in_channels=end_channels, out_channels=out_channels, kernel_size=(1, 1)), )

        self.clf = nn.Linear(num_nodes, class_num)

    def forward(self, input):

        scale = input
        adj_matrix = self.gc(torch.arange(self.num_nodes).to(self.device))
        out = []

        for i in range(len(self.kernel_set)):
            output = self.gnet1[i](scale[i], adj_matrix[i]) + self.gnet2[i](scale[i], adj_matrix[i].transpose(1, 0))
            out.append(self.scale_convs[i](output))

        fusion_out = self.msffgate(out)
        pred = self.end_conv(fusion_out).squeeze()
        logits = self.clf(pred)

        return logits
