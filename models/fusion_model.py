import torch
from torch import nn
from models.common import reflect_conv


class CSAM(nn.Module):
    """
    Channel and Spatial Attention Module (CSAM)

    同时建模通道和空间注意力，用于对差分特征 F_vi^i - F_ir^i / F_ir^i - F_vi^i
    进行自适应加权，从而作为模态补偿信息加入到原始特征中。
    """

    def __init__(self, channels, reduction=16, spatial_kernel_size=7):
        super(CSAM, self).__init__()
        # 通道注意力：GAP + GMP -> MLP -> Sigmoid
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)

        mid_channels = max(channels // reduction, 1)
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(channels, mid_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, channels, kernel_size=1, bias=False),
        )

        # 空间注意力：拼接通道平均池化与最大池化 -> Conv -> Sigmoid
        self.spatial_conv = nn.Conv2d(
            in_channels=2,
            out_channels=1,
            kernel_size=spatial_kernel_size,
            padding=spatial_kernel_size // 2,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, C, H, W]
        # ----- Channel attention -----
        gap_feat = self.gap(x)
        gmp_feat = self.gmp(x)
        channel_att = self.channel_mlp(gap_feat) + self.channel_mlp(gmp_feat)
        channel_att = self.sigmoid(channel_att)  # [B, C, 1, 1]

        x_channel = x * channel_att

        # ----- Spatial attention -----
        avg_pool = torch.mean(x_channel, dim=1, keepdim=True)
        max_pool, _ = torch.max(x_channel, dim=1, keepdim=True)
        spatial_att = self.spatial_conv(torch.cat([avg_pool, max_pool], dim=1))
        spatial_att = self.sigmoid(spatial_att)  # [B, 1, H, W]

        x_out = x_channel * spatial_att
        return x_out


def EPIA_CMDAF(vi_feature, ir_feature, csam_module: CSAM):
    """
    Enhanced Progressive Infrared and Visible Image Fusion (EPIAFusion)
    中的差分融合模块：

        F1 = F_vi^i - F_ir^i
        F2 = F_ir^i - F_vi^i

        \hat{F}_{vi}^{i} = F_{ir}^{i} \oplus CSAM(F1)
        \hat{F}_{ir}^{i} = F_{vi}^{i} \oplus CSAM(F2)

    其中 CSAM 同时建模通道与空间注意力，用作差分信息的补偿项。
    """
    # 差分特征
    F1 = vi_feature - ir_feature
    F2 = ir_feature - vi_feature

    # 通过 CSAM 进行通道+空间注意力加权
    comp_vi = csam_module(F1)
    comp_ir = csam_module(F2)

    # 将补偿信息加到对侧模态特征上
    hat_vi = ir_feature + comp_vi
    hat_ir = vi_feature + comp_ir

    return hat_vi, hat_ir


def Fusion(vi_out, ir_out):
    return torch.cat([vi_out, ir_out], dim=1)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.vi_conv1 = nn.Conv2d(in_channels=1, kernel_size=1, out_channels=16, stride=1, padding=0)
        self.ir_conv1 = nn.Conv2d(in_channels=1, kernel_size=1, out_channels=16, stride=1, padding=0)

        self.vi_conv2 = reflect_conv(in_channels=16, kernel_size=3, out_channels=16, stride=1, pad=1)
        self.ir_conv2 = reflect_conv(in_channels=16, kernel_size=3, out_channels=16, stride=1, pad=1)

        self.vi_conv3 = reflect_conv(in_channels=16, kernel_size=3, out_channels=32, stride=1, pad=1)
        self.ir_conv3 = reflect_conv(in_channels=16, kernel_size=3, out_channels=32, stride=1, pad=1)

        self.vi_conv4 = reflect_conv(in_channels=32, kernel_size=3, out_channels=64, stride=1, pad=1)
        self.ir_conv4 = reflect_conv(in_channels=32, kernel_size=3, out_channels=64, stride=1, pad=1)

        self.vi_conv5 = reflect_conv(in_channels=64, kernel_size=3, out_channels=128, stride=1, pad=1)
        self.ir_conv5 = reflect_conv(in_channels=64, kernel_size=3, out_channels=128, stride=1, pad=1)

        # 对应每一层差分特征通道数的 CSAM，用于 EPIAFusion 的差分补偿
        self.csam_2 = CSAM(channels=16)
        self.csam_3 = CSAM(channels=32)
        self.csam_4 = CSAM(channels=64)

    @staticmethod
    def _fusion_stage(vi_feat, ir_feat, csam_module: CSAM):
        """
        单个尺度上的差分融合阶段，使用 EPIA_CMDAF 完成。
        """
        return EPIA_CMDAF(vi_feat, ir_feat, csam_module)

    def forward(self, y_vi_image, ir_image):
        activate = nn.LeakyReLU()
        vi_out = activate(self.vi_conv1(y_vi_image))
        ir_out = activate(self.ir_conv1(ir_image))

        # 第 2 层特征 + 差分 CSAM 融合
        vi_feat2 = activate(self.vi_conv2(vi_out))
        ir_feat2 = activate(self.ir_conv2(ir_out))
        vi_out, ir_out = self._fusion_stage(vi_feat2, ir_feat2, self.csam_2)

        # 第 3 层特征 + 差分 CSAM 融合
        vi_feat3 = activate(self.vi_conv3(vi_out))
        ir_feat3 = activate(self.ir_conv3(ir_out))
        vi_out, ir_out = self._fusion_stage(vi_feat3, ir_feat3, self.csam_3)

        # 第 4 层特征 + 差分 CSAM 融合
        vi_feat4 = activate(self.vi_conv4(vi_out))
        ir_feat4 = activate(self.ir_conv4(ir_out))
        vi_out, ir_out = self._fusion_stage(vi_feat4, ir_feat4, self.csam_4)

        # 编码末端，不再做差分融合，只提取高级特征
        vi_out, ir_out = activate(self.vi_conv5(vi_out)), activate(self.ir_conv5(ir_out))
        return vi_out, ir_out


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.conv1 = reflect_conv(in_channels=256, kernel_size=3, out_channels=256, stride=1, pad=1)
        self.conv2 = reflect_conv(in_channels=256, kernel_size=3, out_channels=128, stride=1, pad=1)
        self.conv3 = reflect_conv(in_channels=128, kernel_size=3, out_channels=64, stride=1, pad=1)
        self.conv4 = reflect_conv(in_channels=64, kernel_size=3, out_channels=32, stride=1, pad=1)
        self.conv5 = nn.Conv2d(in_channels=32, kernel_size=1, out_channels=1, stride=1, padding=0)

    def forward(self, x):
        activate = nn.LeakyReLU()
        x = activate(self.conv1(x))
        x = activate(self.conv2(x))
        x = activate(self.conv3(x))
        x = activate(self.conv4(x))
        x = nn.Tanh()(self.conv5(x)) / 2 + 0.5  # 将范围从[-1,1]转换为[0,1]
        return x


class PIAFusion(nn.Module):
    def __init__(self):
        super(PIAFusion, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, y_vi_image, ir_image):
        vi_encoder_out, ir_encoder_out = self.encoder(y_vi_image, ir_image)
        encoder_out = Fusion(vi_encoder_out, ir_encoder_out)
        fused_image = self.decoder(encoder_out)
        return fused_image
