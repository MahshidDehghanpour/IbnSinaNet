import torch
from torch import nn

from models import layers


class CSPDenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, groups, opt):
        super().__init__()

        if opt.activation == "Relu":
            activation = nn.ReLU(True)
        elif opt.activation == "LeakyRelu":
            activation = nn.LeakyReLU(.03, True)
        else:
            raise "Activation function not recognized"
        self.sequential = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1, groups=groups),
        )
        # self.dropout = None
        # if dropout_percent > 0:
        #     self.dropout = nn.Dropout2d(dropout_percent)

    def forward(self, x):
        out = self.sequential(x)
        # if self.dropout:
        #     out = self.dropout(out)
        return out


class FCDenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, layers_count, csp, opt, concat_input=True, groups=3):
        super().__init__()
        self.csp = csp
        self.growth_rate = growth_rate
        self.in_channels = in_channels
        self.concat_input = concat_input
        self.main_channels = in_channels // 6 * 3 if csp else in_channels
        self.layers = nn.ModuleList()
        for i in range(layers_count):
            self.layers.append(
                CSPDenseLayer(self.main_channels + i * growth_rate, growth_rate, groups, opt))

    def forward(self, x):
        hc = x.size(1) - self.main_channels
        holdout = x[:, :hc, :, :] if self.csp else None
        path = x[:, hc:, :, :] if self.csp else x
        new_features = []
        for layer in self.layers:
            out = layer(path)
            path = torch.cat([path, out], dim=1)
            new_features.append(out)
        if not self.concat_input:
            return torch.cat(new_features, dim=1)
        if self.csp:
            return torch.cat([holdout, path], dim=1)
        return path

    def out_channels(self):
        main_out = self.growth_rate * len(self.layers)
        return self.in_channels + main_out if self.concat_input else main_out


class FCDenseStage(nn.Module):
    def __init__(self, block_count, in_channels, growth_rate, layers_count, csp, opt, decoder):
        super().__init__()
        self.blocks = nn.ModuleList()
        curr_ch = in_channels
        for i in range(block_count):
            block = FCDenseBlock(curr_ch, growth_rate, layers_count, csp, opt=opt, groups=1 if decoder else 3)
            self.blocks.append(block)
            curr_ch = block.out_channels()
        self.sa = layers.SqueezeAttentionBlock(curr_ch, decoder, opt)
        self.out_ch = curr_ch

    def forward(self, x):
        out = x
        for block in self.blocks:
            out = block(out)
        out = self.sa(out)
        return out

    def out_channels(self):
        return self.out_ch
