import torch
import torch.nn as nn
import torch.nn.functional as F
from .GLayer import gconv, ScoreGraph


class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out, k_size):
        super(ConvBlock, self).__init__()

        self.block = nn.Sequential(nn.Conv2d(c_in, c_out, kernel_size=(1, k_size), stride=1),
                                   nn.ReLU(),
                                   )

    def forward(self, input):
        return self.block(input)


class ResiCBlock(nn.Module):
    def __init__(self, c_in, c_out, k_size):
        super(ResiCBlock, self).__init__()
        self.residual = nn.Conv2d(c_in, c_out, kernel_size=(1, 2), stride=1)
        self.block = nn.Sequential(nn.Conv2d(c_in, c_out, kernel_size=(1, k_size), stride=1),
                                   nn.Conv2d(c_out, c_out, 1, (1, 2)),
                                   )

    def forward(self, input):
        x = self.block(input)
        resi = self.residual(input)
        return F.relu(x + resi[..., -x.size(3):])


class PadCBlock(nn.Module):
    def __init__(self, c_in, c_out, k_size):
        super(PadCBlock, self).__init__()
        pad_len = (k_size - 1) // 2
        self.block = nn.Sequential(nn.Conv2d(c_in, c_out, kernel_size=1, stride=1),
                                   nn.Conv2d(c_out, c_out, kernel_size=(1, k_size), padding=(0, pad_len), stride=1),
                                   nn.Conv2d(c_out, c_out, kernel_size=1, stride=1),
                                   nn.BatchNorm2d(c_out),
                                   nn.ReLU(inplace=True),
                                   )

    def forward(self, input):
        return self.block(input)


class AvgChannelFusion(nn.Module):
    def __init__(self, channel, reduction=4):
        super(AvgChannelFusion, self).__init__()
        self.pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        b, c, _, _ = input.size()
        weight = self.pool(input).view(b, c)
        weight = self.fc(weight).view(b, c, 1, 1)
        return input * weight.expand_as(input)


class MSFFusion(nn.Module):
    def __init__(self, skip_channels, f_num, ratio=1):
        super(MSFFusion, self).__init__()

        self.lin1 = nn.Linear(in_features=skip_channels * f_num, out_features=int(f_num * ratio), bias=False)
        self.lin2 = nn.Linear(in_features=int(f_num * ratio), out_features=f_num, bias=False)
        self.pool1 = nn.AdaptiveAvgPool2d(1)

    def forward(self, input):
        in_cat = torch.cat(input, dim=1)
        in_stk = torch.stack(input, dim=1)

        tep = self.pool1(in_cat).squeeze()
        tep = torch.tanh(self.lin1(tep))
        tep = torch.sigmoid(self.lin2(tep))

        tep = tep.view(in_stk.size(0), in_stk.size(1), 1, 1, 1)

        wht_mul = torch.mul(in_stk, tep)
        # return wht_mul.reshape(in_cat.size())
        return wht_mul.flatten(1, 2)


class MSFF(nn.Module):

    def __init__(self, c_in, c_out, k_set):
        super(MSFF, self).__init__()
        self.in_channels = c_in
        self.out_channels = c_out
        self.fnums = len(k_set)
        self.conv0 = nn.Conv2d(1, c_in, 1, 1)
        self.convs = nn.ModuleList([ResiCBlock(c_in, c_out, k_size=k, ) for k in k_set])

    def forward(self, input):
        input = input.transpose(-2, -1).unsqueeze(dim=1) if input.dim() == 3 else input
        out0 = self.conv0(input)
        return [conv(out0) for conv in self.convs]


class MSFFMIX(nn.Module):

    def __init__(self, c_in, c_out, k_set):
        super(MSFFMIX, self).__init__()

        self.convs = nn.ModuleList([PadCBlock(c_in, c_out, k_size=k) for k in k_set])

        self.mix = nn.Sequential(
            nn.Conv2d(len(self.convs) * c_out, c_out, 1, stride=1),
            nn.Conv2d(c_out, c_in, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            nn.BatchNorm2d(c_in),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        conv_out = [conv(input) for conv in self.convs]
        mix_out = self.mix(torch.cat(conv_out, 1))

        return conv_out, mix_out


class BasicConv(nn.Module):
    def __init__(self, c_in, c_out, k_size):
        super(BasicConv, self).__init__()

        self.trend = nn.Sequential(nn.AvgPool2d((1, 2), stride=1),
                                   nn.Conv2d(c_in, c_out, 1, 1))

        self.block = nn.Sequential(nn.Conv2d(c_in, c_out, kernel_size=1, stride=1),
                                   nn.Conv2d(c_out, c_out, kernel_size=(1, k_size), stride=1),
                                   nn.ReLU(inplace=True),
                                   )

    def forward(self, input):
        out1 = self.trend(input)
        out2 = self.block(input)
        return self.relu(out2 + out1[..., :out2.shape[3]])


class MSGFF(nn.Module):
    def __init__(self, c_in, c_out, k_set, nodes, dim, device, f_num, k, alpha):
        super(MSGFF, self).__init__()
        '''
        Graph Structure Feature Fusion
        Parallel Feature Extractor
        '''
        self.gconv = gconv()
        self.GS = ScoreGraph(nodes, dim, device, f_num, k, alpha)

        self.scale = nn.ModuleList()
        self.ChannelFusion = nn.ModuleList()
        self.f_num = f_num
        self.start_conv = nn.Conv2d(c_in, c_out, kernel_size=(1, 1), stride=(1, 1))
        for i in range(len(k_set)):
            self.scale.append(BasicConv(c_out, c_out, k_set[i]))
            self.norm.append(nn.BatchNorm2d(c_out, affine=False))
            self.ChannelFusion.append(AvgChannelFusion(c_out))

        self.gate_fusion = AvgChannelFusion(c_out)

    def forward(self, input):
        res = []
        adj_set = self.GS

        out0 = self.start_conv(input)
        res.append(out0)
        for i in range(len(self.f_num)):
            out = self.scale[i](out0)
            out = self.g_conv(out, adj_set[i])
            res.append(out)

        return res


if __name__ == '__main__':
    c_in = 16
    c_out = 64

    msff = MSFF(c_in, c_out, [7, 6, 3, 2])

    input_tensor = torch.randn(1, 120, 32)

    out = msff(input_tensor)

    print(f"output shape: {out.shape}")
