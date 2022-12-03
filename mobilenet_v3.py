from typing import Callable, List, Optional

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from functools import partial


# 调整channel为8的整数倍
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


# 卷积+BN+激活函数，继承自nn.Sequential
class ConvBNActivation(nn.Sequential):
    def __init__(self,
                 in_planes: int,
                 out_planes: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 groups: int = 1,# groups: 1,普通卷积；in_channel，dw卷积
                 norm_layer: Optional[Callable[..., nn.Module]] = None, # BN层
                 activation_layer: Optional[Callable[..., nn.Module]] = None): # 激活层
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d # 默认BN
        if activation_layer is None:
            activation_layer = nn.ReLU6 # 默认RELU6
        super(ConvBNActivation, self).__init__(nn.Conv2d(in_channels=in_planes,
                                                         out_channels=out_planes,
                                                         kernel_size=kernel_size,
                                                         stride=stride,
                                                         padding=padding,
                                                         groups=groups,
                                                         bias=False), # 卷积
                                               norm_layer(out_planes), # BN
                                               activation_layer(inplace=True)) # 激活

# SE模块（注意力机制模块）
class SqueezeExcitation(nn.Module):
    def __init__(self, input_c: int, squeeze_factor: int = 4): # input_c 输入通道数； squeeze_factor = 4 输入特征通道的1/4
        super(SqueezeExcitation, self).__init__()
        squeeze_c = _make_divisible(input_c // squeeze_factor, 8) # 调整通道为8的整数倍
        self.fc1 = nn.Conv2d(input_c, squeeze_c, 1) # 1*1的卷积相当于全连接，输入input_c，输出1/4 input_c
        self.fc2 = nn.Conv2d(squeeze_c, input_c, 1) # 输入1/4 input_c，输出input_c

    def forward(self, x: Tensor) -> Tensor: # 正向传播的过程
        scale = F.adaptive_avg_pool2d(x, output_size=(1, 1)) # 全局平均池化，将每一个通道特征输出为1*1的特征
        scale = self.fc1(scale) # 1*1卷积
        scale = F.relu(scale, inplace=True) # relu激活
        scale = self.fc2(scale) # 1*1卷积
        scale = F.hardsigmoid(scale, inplace=True) # hardsigmoid激活
        return scale * x # 每一个通道特征 * 通道权重


# 倒残差结构配置
class InvertedResidualConfig:
    def __init__(self,
                 input_c: int,
                 kernel: int,
                 expanded_c: int,
                 out_c: int,
                 use_se: bool,
                 activation: str,
                 stride: int,
                 width_multi: float): # α参数
        self.input_c = self.adjust_channels(input_c, width_multi)
        self.kernel = kernel
        self.expanded_c = self.adjust_channels(expanded_c, width_multi)
        self.out_c = self.adjust_channels(out_c, width_multi)
        self.use_se = use_se
        self.use_hs = activation == "HS"  # Hardswish 或者 Relu（默认）
        self.stride = stride

    # 调整channel
    @staticmethod
    def adjust_channels(channels: int, width_multi: float):
        return _make_divisible(channels * width_multi, 8)

# 倒残差结构，继承自nn.Module
class InvertedResidual(nn.Module):
    def __init__(self,
                 cnf: InvertedResidualConfig, # 倒残差配置
                 norm_layer: Callable[..., nn.Module]): # BN
        super(InvertedResidual, self).__init__()

        if cnf.stride not in [1, 2]: # stride 只能为1和2
            raise ValueError("illegal stride value.")

        # short_cut连接，只有当stride=1以及输入输出channal一致的情况下才用
        self.use_res_connect = (cnf.stride == 1 and cnf.input_c == cnf.out_c)

        layers: List[nn.Module] = []
        #  Hardswish 或者 Relu（默认）pytorch 1.7以上才有Hardswish
        activation_layer = nn.Hardswish if cnf.use_hs else nn.ReLU

        # 1* 1升维：只有expanded channel和input channel相等
        if cnf.expanded_c != cnf.input_c:
            layers.append(ConvBNActivation(cnf.input_c,
                                           cnf.expanded_c,
                                           kernel_size=1,
                                           norm_layer=norm_layer,
                                           activation_layer=activation_layer))

        # 深度卷积：depthwise
        layers.append(ConvBNActivation(cnf.expanded_c,
                                       cnf.expanded_c,
                                       kernel_size=cnf.kernel,
                                       stride=cnf.stride,
                                       groups=cnf.expanded_c, # groups 数跟 expanded channel保持一致
                                       norm_layer=norm_layer,
                                       activation_layer=activation_layer))

        # 是否使用se模块
        if cnf.use_se:
            layers.append(SqueezeExcitation(cnf.expanded_c))

        # 1 * 1 降维
        layers.append(ConvBNActivation(cnf.expanded_c,
                                       cnf.out_c,
                                       kernel_size=1,
                                       norm_layer=norm_layer,
                                       # 线性激活
                                       activation_layer=nn.Identity))

        self.block = nn.Sequential(*layers)
        self.out_channels = cnf.out_c
        self.is_strided = cnf.stride > 1

    def forward(self, x: Tensor) -> Tensor:
        result = self.block(x)
        if self.use_res_connect: # 使用short_cut连接
            result += x

        return result


class MobileNetV3(nn.Module):
    def __init__(self,
                 # 倒残差参数列表
                 inverted_residual_setting: List[InvertedResidualConfig],
                 last_channel: int,
                 # 分类类别个数
                 num_classes: int = 1000,
                 block: Optional[Callable[..., nn.Module]] = None,
                 norm_layer: Optional[Callable[..., nn.Module]] = None):
        super(MobileNetV3, self).__init__()

        # 倒残差参数列表检查
        if not inverted_residual_setting:
            raise ValueError("The inverted_residual_setting should not be empty.")
        elif not (isinstance(inverted_residual_setting, List) and
                  all([isinstance(s, InvertedResidualConfig) for s in inverted_residual_setting])):
            raise TypeError("The inverted_residual_setting should be List[InvertedResidualConfig]")

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)

        layers: List[nn.Module] = []

        # building first layer
        firstconv_output_c = inverted_residual_setting[0].input_c
        layers.append(ConvBNActivation(3,
                                       firstconv_output_c,
                                       kernel_size=3,
                                       stride=2,
                                       norm_layer=norm_layer,
                                       activation_layer=nn.Hardswish))
        # building inverted residual blocks
        for cnf in inverted_residual_setting:
            layers.append(block(cnf, norm_layer))

        # building last several layers
        lastconv_input_c = inverted_residual_setting[-1].out_c
        lastconv_output_c = 6 * lastconv_input_c
        layers.append(ConvBNActivation(lastconv_input_c,
                                       lastconv_output_c,
                                       kernel_size=1,
                                       norm_layer=norm_layer,
                                       activation_layer=nn.Hardswish))
        # 第一个卷积到池化之前的卷积添加到layers中
        self.features = nn.Sequential(*layers)
        # 平均池化
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # 倒数第二个全连接 + hs + droupout + 最后一个全连接
        self.classifier = nn.Sequential(nn.Linear(lastconv_output_c, last_channel),
                                        nn.Hardswish(inplace=True),
                                        nn.Dropout(p=0.2, inplace=True),
                                        nn.Linear(last_channel, num_classes))

        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.features(x) # 平均池化之前
        x = self.avgpool(x) # 平均池化
        x = torch.flatten(x, 1) # 展平处理，把1*1的高和宽维度舍弃掉，展平成1维
        x = self.classifier(x) # 两个全连接

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def mobilenet_v3_large(num_classes: int = 1000,
                       reduced_tail: bool = False) -> MobileNetV3:
    """
    Constructs a large MobileNetV3 architecture from
    "Searching for MobileNetV3" <https://arxiv.org/abs/1905.02244>.

    weights_link:
    https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth

    Args:
        num_classes (int): number of classes
        reduced_tail (bool): If True, reduces the channel counts of all feature layers
            between C4 and C5 by 2. It is used to reduce the channel redundancy in the
            backbone for Detection and Segmentation.
    """
    width_multi = 1.0 # α参数
    bneck_conf = partial(InvertedResidualConfig, width_multi=width_multi)
    adjust_channels = partial(InvertedResidualConfig.adjust_channels, width_multi=width_multi)

    reduce_divider = 2 if reduced_tail else 1

    inverted_residual_setting = [
        # input_c, kernel, expanded_c, out_c, use_se, activation, stride
        bneck_conf(16, 3, 16, 16, False, "RE", 1),
        bneck_conf(16, 3, 64, 24, False, "RE", 2),  # C1
        bneck_conf(24, 3, 72, 24, False, "RE", 1),
        bneck_conf(24, 5, 72, 40, True, "RE", 2),  # C2
        bneck_conf(40, 5, 120, 40, True, "RE", 1),
        bneck_conf(40, 5, 120, 40, True, "RE", 1),
        bneck_conf(40, 3, 240, 80, False, "HS", 2),  # C3
        bneck_conf(80, 3, 200, 80, False, "HS", 1),
        bneck_conf(80, 3, 184, 80, False, "HS", 1),
        bneck_conf(80, 3, 184, 80, False, "HS", 1),
        bneck_conf(80, 3, 480, 112, True, "HS", 1),
        bneck_conf(112, 3, 672, 112, True, "HS", 1),
        bneck_conf(112, 5, 672, 160 // reduce_divider, True, "HS", 2),  # C4
        bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1),
        bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1),
    ]
    last_channel = adjust_channels(1280 // reduce_divider)  # C5

    return MobileNetV3(inverted_residual_setting=inverted_residual_setting,
                       last_channel=last_channel,
                       num_classes=num_classes)


def mobilenet_v3_small(num_classes: int = 1000,
                       reduced_tail: bool = False) -> MobileNetV3:
    """
    Constructs a large MobileNetV3 architecture from
    "Searching for MobileNetV3" <https://arxiv.org/abs/1905.02244>.

    weights_link:
    https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth

    Args:
        num_classes (int): number of classes
        reduced_tail (bool): If True, reduces the channel counts of all feature layers
            between C4 and C5 by 2. It is used to reduce the channel redundancy in the
            backbone for Detection and Segmentation.
    """
    width_multi = 1.0
    bneck_conf = partial(InvertedResidualConfig, width_multi=width_multi)
    adjust_channels = partial(InvertedResidualConfig.adjust_channels, width_multi=width_multi)

    reduce_divider = 2 if reduced_tail else 1

    inverted_residual_setting = [
        # input_c, kernel, expanded_c, out_c, use_se, activation, stride
        bneck_conf(16, 3, 16, 16, True, "RE", 2),  # C1
        bneck_conf(16, 3, 72, 24, False, "RE", 2),  # C2
        bneck_conf(24, 3, 88, 24, False, "RE", 1),
        bneck_conf(24, 5, 96, 40, True, "HS", 2),  # C3
        bneck_conf(40, 5, 240, 40, True, "HS", 1),
        bneck_conf(40, 5, 240, 40, True, "HS", 1),
        bneck_conf(40, 5, 120, 48, True, "HS", 1),
        bneck_conf(48, 5, 144, 48, True, "HS", 1),
        bneck_conf(48, 5, 288, 96 // reduce_divider, True, "HS", 2),  # C4
        bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1),
        bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1)
    ]
    last_channel = adjust_channels(1024 // reduce_divider)  # C5

    return MobileNetV3(inverted_residual_setting=inverted_residual_setting,
                       last_channel=last_channel,
                       num_classes=num_classes)