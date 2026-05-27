"""
tiramisu.py — نسخه آپدیت شده برای ablation study
تغییرات:
  1. first_conv: اگه ablation_gcnn=False باشه، groups=1 (بدون GCNN)
  2. encoder stages: اگه ablation_csp=False باشه، csp=False در همه stages
"""
import torch.nn as nn
from models import layers
from models.blocks import FCDenseStage


def get_middle_channels(channels):
    chunk = channels // 3
    return chunk, 2 * chunk, chunk, channels


class FCDenseNet(nn.Module):
    def __init__(self, in_channels, down_layers, down_blocks, up_layers, up_blocks,
                 growth_rate, first_channels, n_classes, opt):
        super().__init__()

        self.down_layers = down_layers
        self.down_blocks = down_blocks
        self.up_layers   = up_layers
        self.up_blocks   = up_blocks
        self.skip_channels = []

        # ── تغییر ۱: GCNN ────────────────────────────────────────────────
        # مدل اصلی: groups=3  (یعنی GCNN فعاله)
        # ablation بدون GCNN: groups=1 (conv معمولی)
        use_gcnn = getattr(opt, 'ablation_gcnn', True)
        gcnn_groups = 3 if use_gcnn else 1
        self.first_conv = nn.Conv2d(
            in_channels=in_channels, out_channels=first_channels,
            kernel_size=3, padding=1, groups=gcnn_groups)
        # ─────────────────────────────────────────────────────────────────

        curr_ch = first_channels

        # ── تغییر ۲: CSP ─────────────────────────────────────────────────
        # مدل اصلی: csp=True در encoder (به جز آخرین stage)
        # ablation بدون CSP: csp=False همه جا
        use_csp = getattr(opt, 'ablation_csp', True)
        # ─────────────────────────────────────────────────────────────────

        self.encoder_stages    = nn.ModuleList([])
        self.encoder_downscales = nn.ModuleList([])

        for idx, block_count in enumerate(self.down_blocks):
            last = idx == len(self.down_blocks) - 1

            # ── تغییر ۲ اعمال می‌شه اینجا ─────────────────────────────
            if not last:
                csp_flag = True if use_csp else False
            else:
                csp_flag = False   # آخرین stage هیچوقت CSP نداره
            # ─────────────────────────────────────────────────────────────

            stage = FCDenseStage(down_blocks[idx], curr_ch, growth_rate,
                                 down_layers[idx], csp=csp_flag, opt=opt,
                                 decoder=False)
            curr_ch = stage.out_channels()
            self.encoder_stages.append(stage)

            if not last:
                self.skip_channels.insert(0, get_middle_channels(curr_ch))
                self.encoder_downscales.append(layers.TransitionDown(curr_ch))

        self.decoder_stages   = nn.ModuleList([])
        self.decoder_upscales = nn.ModuleList([])

        for idx, block_count in enumerate(self.up_blocks):
            self.decoder_upscales.append(layers.TransitionUp())
            curr_ch = curr_ch + self.skip_channels[idx][2]
            stage = FCDenseStage(up_blocks[idx], curr_ch, growth_rate,
                                 up_layers[idx], csp=False, opt=opt,
                                 decoder=True)
            curr_ch = stage.out_channels()
            self.decoder_stages.append(stage)

        self.finalConv = nn.Conv2d(in_channels=curr_ch, out_channels=n_classes,
                                   kernel_size=1, padding=0)

    def forward(self, x):
        out = self.first_conv(x)
        skip_connections = []
        for idx, stage in enumerate(self.encoder_stages):
            out = stage(out)
            if idx < len(self.encoder_downscales):
                skip_connections.append(out)
                out = self.encoder_downscales[idx](out)

        for idx, stage in enumerate(self.decoder_stages):
            skip       = skip_connections.pop()
            skip_slice = self.skip_channels[idx]
            out = self.decoder_upscales[idx](
                skip[:, skip_slice[0]:skip_slice[1], :, :], out)
            out = stage(out)

        out = self.finalConv(out)
        return out


def FCDenseNet67(n_classes, grow_rate, opt):
    return FCDenseNet(
        in_channels=3,
        down_layers=opt.down_layers, down_blocks=opt.down_blocks,
        up_layers=opt.up_layers,     up_blocks=opt.up_blocks,
        growth_rate=grow_rate,       first_channels=48,
        n_classes=n_classes,         opt=opt)
