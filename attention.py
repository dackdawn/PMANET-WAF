import torch.nn as nn

class CBAMLayer(nn.Module):
    """
        Initializes a CBAM (Attention Module) Layer.
        CBAM是一个轻量级的注意力模块，用于增强深度神经网络的表示能力。这里实现的是其中的通道注意力（Channel Attention）部分。
        :param channel: The number of input channels.
        :param reduction: The reduction ratio for channel attention (default is 3).
        :param spatial_kernel: The size of the spatial kernel (default is 7).

        It's worth noting that we only utilize the channel attention in CBAM.
    """
    def __init__(self, channel, reduction=3, spatial_kernel=7):
        super(CBAMLayer, self).__init__()

        # 池化操作：将H和W压缩到1
        # Channel attention: Compress H and W to 1
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # 共享MLP (多层感知机)
        # 使用1x1卷积代替全连接层
        # 包含降维->ReLU激活->升维的过程
        # Shared MLP
        self.mlp = nn.Sequential(
            # Use Conv2d for more convenient operations compared to Linear

            # 降维 输入: [B, C, 1, 1] -> 输出: [B, C//reduction, 1, 1]
            nn.Conv2d(channel, channel // reduction, 1, bias=False),

            # 维度保持不变: [B, C//reduction, 1, 1]
            nn.ReLU(inplace=True),  # Inplace operation to save memory
 
            # 升维 输入: [B, C//reduction, 1, 1] -> 输出: [B, C, 1, 1]
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
           输入特征 [B,C,H,W]
            ↙            ↘
        最大池化            平均池化
        [B,C,1,1]         [B,C,1,1]
            ↓                ↓
        MLP              MLP
            ↓                ↓
            --------+--------
                    ↓
                Sigmoid
                    ↓
            通道注意力权重
                    ↓
            特征加权(乘法)
                    ↓
            输出特征
        """

        # x: [batch_size, channel, height, width]

        # max_pool 输入: [B, C, H, W] -> 输出: [B, C, 1, 1]
        # mlp 输入: [B, C, 1, 1] -> 输出: [B, C, 1, 1]
        max_out = self.mlp(self.max_pool(x))

        # avg_pool 输入: [B, C, H, W] -> 输出: [B, C, 1, 1]
        # mlp 输入: [B, C, 1, 1] -> 输出: [B, C, 1, 1]
        avg_out = self.mlp(self.avg_pool(x))

        # 输入: [B, C, 1, 1] + [B, C, 1, 1] -> 输出: [B, C, 1, 1]
        channel_out = self.sigmoid(max_out + avg_out)

        # channel_out: [B, C, 1, 1] (会自动广播)
        # x: [B, C, H, W]
        # 输出: [B, C, H, W]
        x = channel_out * x

        return x
