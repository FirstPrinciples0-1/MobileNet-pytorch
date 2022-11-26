from torch import nn
import torch


def _make_divisible(ch, divisor=8, min_ch=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch

## 卷积/BN/Relu6，继承自nn.Sequential
class ConvBNReLU(nn.Sequential):
    # groups : 1,普通卷积；in_channel，dw卷积
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1):
        # same卷积
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            # 不使用偏置，因为BN，偏置没有用
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU6(inplace=True)
        )


# 倒残差结构（两头channel小，中间channel大），继承自nn.Module
class InvertedResidual(nn.Module):
    # 扩展因子expand_ratio ： t
    def __init__(self, in_channel, out_channel, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        # hidden_channel： tk
        hidden_channel = in_channel * expand_ratio
        # 是否使用shortcut分支：当stride = 1 且输入输出的channel相同才使用
        self.use_shortcut = stride == 1 and in_channel == out_channel

        layers = []
        # 当扩展因子不为1
        if expand_ratio != 1:
            # 1x1 pointwise conv
            layers.append(ConvBNReLU(in_channel, hidden_channel, kernel_size=1))
        layers.extend([
            # 3x3 depthwise conv
            ConvBNReLU(hidden_channel, hidden_channel, stride=stride, groups=hidden_channel),
            # 1x1 pointwise conv(linear)
            # 不添加激活函数，就相当于linear激活函数（y=x）
            nn.Conv2d(hidden_channel, out_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel),
        ])

        # *layers位置参数的形式传入
        self.conv = nn.Sequential(*layers)

    # 正向传播过程
    def forward(self, x):
        if self.use_shortcut:
            return x + self.conv(x)
        else:
            return self.conv(x)

# 模型，继承自nn.Module
class MobileNetV2(nn.Module):
    # alpha 超参，控制channal的倍率
    def __init__(self, num_classes=1000, alpha=1.0, round_nearest=8):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        # channel设置为8的整数倍，猜测跟CPU或者GPU的并行处理有关
        input_channel = _make_divisible(32 * alpha, round_nearest)
        last_channel = _make_divisible(1280 * alpha, round_nearest)

        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        features = []
        # conv1 layer
        features.append(ConvBNReLU(3, input_channel, stride=2))
        # building inverted residual blockes
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * alpha, round_nearest)
            for i in range(n):
                # 如果第一层（i==0），则stride=s，其他stride=1
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU(input_channel, last_channel, 1))
        # 组合成特征层
        self.features = nn.Sequential(*features)

        # building classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            # 全连接层
            nn.Linear(last_channel, num_classes)
        )

        # 权重初始化
        for m in self.modules():
            # 卷积层
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    # 存在偏置，则设置为0
                    nn.init.zeros_(m.bias)
            # BN层
            elif isinstance(m, nn.BatchNorm2d):
                # 方差1
                nn.init.ones_(m.weight)
                # 均值0
                nn.init.zeros_(m.bias)
            # 全连接层
            elif isinstance(m, nn.Linear):
                # 正态分布函数，权重调整为均值为0，方差为0.01的正态分布
                nn.init.normal_(m.weight, 0, 0.01)
                # 偏置设置为0
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x