import torch
import torch.nn as nn
import torch.nn.functional as F


# class DenseLayer(nn.Module):
#     def __init__(self, in_channels, growth_rate, seq_size):
#         super().__init__()
#         self.sequential = nn.Sequential(
#             nn.BatchNorm2d(in_channels),
#             nn.ReLU(True),
#             nn.Conv2d(in_channels, growth_rate, kernel_size=3,
#                       padding=1, groups=seq_size),
#             nn.Dropout2d(0.2)
#         )
#
#     def forward(self, x):
#         return self.sequential(x)


# class DenseBlock(nn.Module):
#     def __init__(self, in_channels, growth_rate, n_layers, upsample=False, seq_size=3, csp=True):
#         super().__init__()
#         self.upsample = upsample
#         self.csp = csp
#         self.main_channels = in_channels // 6 * 3 if csp else in_channels
#         self.layers = nn.ModuleList(
#             [DenseLayer(self.main_channels + i * growth_rate * seq_size, growth_rate * seq_size, seq_size)
#              for i in range(n_layers)])
#
#     def forward(self, x):
#         if self.csp:
#             hc = x.size(1) - self.main_channels
#             holdout = x[:, :hc, :, :]
#             path = x[:, hc:, :, :]
#         else:
#             holdout = None
#             path = x
#         if self.upsample:
#             new_features = []
#             for layer in self.layers:
#                 out = layer(path)
#                 path = torch.cat([path, out], 1)
#                 new_features.append(out)
#             return torch.cat(new_features, 1)
#         else:
#             for layer in self.layers:
#                 out = layer(path)
#                 path = torch.cat([path, out], 1)
#             if self.csp:
#                 return torch.cat([holdout, path], dim=1)
#             return path


class TransitionDown(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.sequential = nn.Sequential(
            nn.BatchNorm2d(num_features=in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x):
        return self.sequential(x)


class TransitionUp(nn.Module):
    def forward(self, skip, x):
        out = F.interpolate(x, (skip.size(2), skip.size(3)))
        out = torch.cat([skip, out], 1)
        return out


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out, decoder, group_multi, dropout_percent):
        super(conv_block, self).__init__()
        groups = 2 if decoder else 4
        groups *= group_multi
        self.conv = nn.Sequential(
            nn.BatchNorm2d(ch_out),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1, groups=groups),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(ch_out),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1, groups=groups),
            nn.ReLU(inplace=True)
        )
        self.dropout = None
        if dropout_percent > 0:
            self.dropout = nn.Dropout2d(dropout_percent)

    def forward(self, x):
        x = self.conv(x)
        if self.dropout:
            x=self.dropout(x)
        return x


class SqueezeAttentionBlock(nn.Module):
    def __init__(self, ch_in, decoder, opt):
        super(SqueezeAttentionBlock, self).__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv = conv_block(ch_in, ch_in, decoder, opt.group_multi, opt.dropout)
        self.conv_atten = conv_block(ch_in, ch_in, decoder, opt.group_multi, 0)


    def forward(self, x):
        x_res = self.conv(x)
        y = self.avg_pool(x)
        y = self.conv_atten(y)
        y = F.interpolate(y, (x_res.size(2), x_res.size(3)))
        return (y * x_res) + y


def center_crop(layer, max_height, max_width):
    _, _, _, h, w = layer.size()
    xy1 = (w - max_width) // 2
    xy2 = (h - max_height) // 2
    return layer[:, :, :, xy2:(xy2 + max_height), xy1:(xy1 + max_width)]
