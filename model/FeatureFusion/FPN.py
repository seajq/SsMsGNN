import torch
import torch.nn as nn
import torch.nn.functional as F


class PadConv(nn.Module):
    def __init__(self, c_in, c_out, k_size):
        super(PadConv, self).__init__()
        pad_len = (k_size - 1) // 2

        self.trend = nn.Sequential(nn.AvgPool2d((1, 2), stride=1),
                                   nn.Conv2d(c_in, c_out, 1, 1))

        self.block = nn.Sequential(nn.Conv2d(c_in, c_out, kernel_size=1, stride=(1, 1)),
                                   nn.BatchNorm2d(c_out, affine=False),
                                   nn.Conv2d(c_out, c_out, kernel_size=(1, k_size), padding=(0, pad_len), stride=1),
                                   nn.BatchNorm2d(c_out, affine=False),
                                   nn.ReLU(inplace=True),
                                   )

    def forward(self, input):
        _, _, dim, win = input.size()
        out1 = self.trend(input)
        out1 = F.interpolate(out1, size=(dim, win), mode='bilinear')
        out2 = self.block(input)
        return self.relu(out2 + out1)


class UniConv(nn.Module):
    def __init__(self, c_in, c_out, k_size):
        super(UniConv, self).__init__()

        self.trend = nn.Sequential(nn.AvgPool2d((1, 2), stride=1),
                                   nn.Conv2d(c_in, c_out, 1, 1))

        self.block = nn.Sequential(nn.Conv2d(c_in, c_out, kernel_size=1, stride=(1, 2)),
                                   nn.BatchNorm2d(c_out, affine=False),
                                   nn.Conv2d(c_out, c_out, kernel_size=(1, k_size), stride=1),
                                   nn.BatchNorm2d(c_out, affine=False),
                                   nn.ReLU(inplace=True),
                                   )

    def forward(self, input):
        out1 = self.trend(input)
        out2 = self.block(input)
        return self.relu(out2 + out1[..., :out2.shape[3]])


class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out, k_size):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=(1, 1), stride=(1, 2))
        self.conv1 = nn.Conv2d(c_in, c_out, kernel_size=(1, k_size), stride=(1, 1))
        self.pool1 = nn.AvgPool2d(kernel_size=(1, 3), stride=(1, 2))
        self.relu = nn.ReLU()

    def forward(self, input):
        conv = self.conv(input)
        conv1 = self.conv1(input)
        pool1 = self.pool1(conv1)
        return self.relu(pool1 + conv[..., -pool1.shape[3]:])


class FPNFusion(nn.Module):
    def __init__(self, skip_channels, f_num, ratio=1):
        super(FPNFusion, self).__init__()

        self.dense1 = nn.Linear(in_features=skip_channels * f_num, out_features=f_num * ratio, bias=False)

        self.dense2 = nn.Linear(in_features=f_num * ratio, out_features=f_num, bias=False)

    def forward(self, input):
        out0 = torch.cat(input, dim=1)
        out1 = torch.stack(input, dim=1)

        se = torch.mean(out0, dim=2, keepdim=False)
        se = torch.squeeze(se)

        se = F.relu(self.dense1(se))
        se = torch.sigmoid(self.dense2(se))

        se = torch.unsqueeze(se, -1)
        se = torch.unsqueeze(se, -1)
        se = torch.unsqueeze(se, -1)

        x = torch.mul(out1, se)
        x = torch.mean(x, dim=1, keepdim=False)
        return x


class MultiScaleConv(nn.Module):
    def __init__(self, c_in, c_out, seq_length, kernel_set):
        super(MultiScaleConv, self).__init__()

        self.seq_length = seq_length
        self.scale = nn.ModuleList()
        self.kernel_set = kernel_set
        self.start_conv = nn.Conv2d(c_in, c_out, kernel_size=(1, 1), stride=(1, 1))

        self.scale.append(nn.Conv2d(c_out, c_out, kernel_size=(1, kernel_set[0]), stride=(1, 1)))

        for i in range(1, len(kernel_set)):
            self.scale.append(ConvBlock(c_out, c_out, kernel_set[i]))

    def forward(self, input):

        scale = []
        scale_temp = input

        scale_temp = self.start_conv(scale_temp)

        for idx, scale_layer in enumerate(self.scale):
            scale_temp = scale_layer(scale_temp)
            scale.append(scale_temp)

        return scale


class FPN(nn.Module):

    def __init__(self, c_in, c_set, k_set):
        super().__init__()
        '''
        Feature Pyramid Network
        Unidirectional Feature Extractor
        '''
        self.conv0 = nn.Conv2d(c_in, c_in, kernel_size=(1, 1), stride=(1, 1))
        self.down = nn.ModuleList()
        self.up = nn.ModuleList()
        self.norm = nn.ModuleList()
        self.ChannelConsist = nn.ModuleList()
        for i in range(len(k_set)):

            if i == 0:
                self.down.append(nn.Conv2d(c_in, c_set[i], kernel_size=(1, k_set[i]), stride=(1, 1)))
                self.norm.append(nn.BatchNorm2d(c_set[i], affine=False))
                self.up.append(nn.ConvTranspose2d(c_set[len(c_set) - 1 - i], c_set[len(k_set) - 2], kernel_size=(1, k_set[len(c_set) - 1 - i]), stride=(1, 1)))
                # self.up.append(nn.ConvTranspose2d(c_set[len(c_set) - 1 - i], c_set[-1], kernel_size=(1, k_set[len(c_set) - 1 - i]), stride=(1, 1)))

                self.ChannelConsist.append(nn.Conv2d(c_set[len(c_set) - 1 - i], c_set[-1], kernel_size=(1, 1), stride=(1, 1)))
            else:
                self.down.append(nn.Conv2d(c_set[i - 1], c_set[i], kernel_size=(1, k_set[i]), stride=(1, 1)))
                self.norm.append(nn.BatchNorm2d(c_set[i], affine=False))
                self.up.append(nn.ConvTranspose2d(c_set[len(c_set) - 1 - i], c_set[len(k_set) - 2 - i], kernel_size=(1, k_set[len(c_set) - 1 - i]), stride=(1, 1)))
                # self.up.append(nn.ConvTranspose2d(c_set[len(c_set) - 1 - i], c_set[-1], kernel_size=(1, k_set[len(c_set) - 1 - i]), stride=(1, 1)))

                self.ChannelConsist.append(nn.Conv2d(c_set[len(c_set) - 1 - i], c_set[-1], kernel_size=(1, 1), stride=(1, 1)))

    def forward(self, input):
        x = self.conv0(input)
        down_out = []
        up_out = []
        res = []
        for i in range(len(self.down)):
            down_out.append(self.down[i](x))
            x = self.down[i](x)
        for j in range(len(self.up) - 1):
            up_out.append(self.up[j](down_out[len(down_out) - 1 - j]))
            res_temp = down_out[len(down_out) - j - 2] + up_out[j]
            res_temp = self.ChannelConsist[j + 1](res_temp)
            res.append(res_temp)

        return [down_out[-1]] + res
        # return res[::-1] + [down_out[-1]]

# if __name__ == '__main__':
#     c_set = [16, 32, 64, 128]
#     k_set = [9, 7, 5, 3]
#     c_in = 3
#     c_out = 256
#
#     fpn = FPN(c_in, c_set, k_set)
#
#     input_tensor = torch.randn(1, c_in, 32, 120)
#
#     out = fpn(input_tensor)
#
#     print(f"Upsampled output shape: {[out.shape for out in out]}")
#     length_set = []
#     # length_set.append(120 - k_set[0] + 1)
#     length = 120
#     for i in range(len(k_set)):
#         length_set.append(int((length - k_set[i]) + 1))
#         length = length_set[i]
#     print(length_set)
