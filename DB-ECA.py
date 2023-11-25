# copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
from paddle import nn
import paddle.nn.functional as F
from paddle import ParamAttr
import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '../../..')))

from ppocr.modeling.backbones.det_mobilenet_v3 import SEModule
from ppocr.modeling.backbones.det_resnet_vd import ResNet_vd, ConvBNLayer, BottleneckBlock, BasicBlock
# from ppocr.modeling.backbones.ECAnet import EfficientChannelAttention


class eca_layer(nn.Layer):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
    """

    def __init__(self, channel, gamma=2, b=1):
        super(eca_layer, self).__init__()

        t = int(abs((np.log2(channel) + b) / gamma))
        k_size = t if t % 2 else t + 1

        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.conv = nn.Conv1D(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias_attr=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  # shape=1,16,40,40
        y = self.avg_pool(x)  # shape=1,16,1,1

        y = y.squeeze(-1)  # shape=1,16,1
        y = paddle.transpose(y, [0, 2, 1])  # shape=1,1,16
        y = self.conv(y)  # shape=1,1,16
        y = paddle.transpose(y, [0, 2, 1])  # shape=1,16,1
        y = y.unsqueeze(-1)  # shape=1,16,1,1

        y = self.sigmoid(y)

        return x * y.expand_as(x)

class DSConv(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding,
                 stride=1,
                 groups=None,
                 if_act=True,
                 act="relu",
                 **kwargs):
        super(DSConv, self).__init__()
        if groups == None:
            groups = in_channels
        self.if_act = if_act
        self.act = act
        self.conv1 = nn.Conv2D(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias_attr=False)

        self.bn1 = nn.BatchNorm(num_channels=in_channels, act=None)

        self.conv2 = nn.Conv2D(
            in_channels=in_channels,
            out_channels=int(in_channels * 4),
            kernel_size=1,
            stride=1,
            bias_attr=False)

        self.bn2 = nn.BatchNorm(num_channels=int(in_channels * 4), act=None)

        self.conv3 = nn.Conv2D(
            in_channels=int(in_channels * 4),
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            bias_attr=False)
        self._c = [in_channels, out_channels]
        if in_channels != out_channels:
            self.conv_end = nn.Conv2D(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                bias_attr=False)

    def forward(self, inputs):

        x = self.conv1(inputs)
        x = self.bn1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        if self.if_act:
            if self.act == "relu":
                x = F.relu(x)
            elif self.act == "hardswish":
                x = F.hardswish(x)
            else:
                print("The activation function({}) is selected incorrectly.".
                      format(self.act))
                exit()

        x = self.conv3(x)
        if self._c[0] != self._c[1]:
            x = x + self.conv_end(inputs)
        return x


class DBFPN(nn.Layer):
    def __init__(self, in_channels, out_channels, use_asf=False, **kwargs):
        super(DBFPN, self).__init__()
        self.out_channels = out_channels
        self.use_asf = use_asf

        weight_attr = paddle.nn.initializer.KaimingUniform()

        self.in2_conv = nn.Conv2D(
            in_channels=in_channels[0],
            out_channels=self.out_channels,
            kernel_size=1,
            weight_attr=ParamAttr(initializer=weight_attr),
            bias_attr=False)
        self.in3_conv = nn.Conv2D(
            in_channels=in_channels[1],
            out_channels=self.out_channels,
            kernel_size=1,
            weight_attr=ParamAttr(initializer=weight_attr),
            bias_attr=False)
        self.in4_conv = nn.Conv2D(
            in_channels=in_channels[2],
            out_channels=self.out_channels,
            kernel_size=1,
            weight_attr=ParamAttr(initializer=weight_attr),
            bias_attr=False)
        self.in5_conv = nn.Conv2D(
            in_channels=in_channels[3],
            out_channels=self.out_channels,
            kernel_size=1,
            weight_attr=ParamAttr(initializer=weight_attr),
            bias_attr=False)
        self.p5_conv = nn.Conv2D(
            in_channels=self.out_channels,
            out_channels=self.out_channels // 4,
            kernel_size=3,
            padding=1,
            weight_attr=ParamAttr(initializer=weight_attr),
            bias_attr=False)
        self.p4_conv = nn.Conv2D(
            in_channels=self.out_channels,
            out_channels=self.out_channels // 4,
            kernel_size=3,
            padding=1,
            weight_attr=ParamAttr(initializer=weight_attr),
            bias_attr=False)
        self.p3_conv = nn.Conv2D(
            in_channels=self.out_channels,
            out_channels=self.out_channels // 4,
            kernel_size=3,
            padding=1,
            weight_attr=ParamAttr(initializer=weight_attr),
            bias_attr=False)
        self.p2_conv = nn.Conv2D(
            in_channels=self.out_channels,
            out_channels=self.out_channels // 4,
            kernel_size=3,
            padding=1,
            weight_attr=ParamAttr(initializer=weight_attr),
            bias_attr=False)

        if self.use_asf is True:
            self.asf = ASFBlock(self.out_channels, self.out_channels // 4)

    def forward(self, x):
        c2, c3, c4, c5 = x

        in5 = self.in5_conv(c5)
        in4 = self.in4_conv(c4)
        in3 = self.in3_conv(c3)
        in2 = self.in2_conv(c2)

        out4 = in4 + F.upsample(
            in5, scale_factor=2, mode="nearest", align_mode=1)  # 1/16
        out3 = in3 + F.upsample(
            out4, scale_factor=2, mode="nearest", align_mode=1)  # 1/8
        out2 = in2 + F.upsample(
            out3, scale_factor=2, mode="nearest", align_mode=1)  # 1/4

        p5 = self.p5_conv(in5)
        p4 = self.p4_conv(out4)
        p3 = self.p3_conv(out3)
        p2 = self.p2_conv(out2)
        p5 = F.upsample(p5, scale_factor=8, mode="nearest", align_mode=1)
        p4 = F.upsample(p4, scale_factor=4, mode="nearest", align_mode=1)
        p3 = F.upsample(p3, scale_factor=2, mode="nearest", align_mode=1)

        fuse = paddle.concat([p5, p4, p3, p2], axis=1)

        if self.use_asf is True:
            fuse = self.asf(fuse, [p5, p4, p3, p2])

        return fuse


class RECALayer(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, shortcut=True):
        super(RSELayer, self).__init__()
        weight_attr = paddle.nn.initializer.KaimingUniform()
        self.out_channels = out_channels
        self.in_conv = nn.Conv2D(
            in_channels=in_channels,
            out_channels=self.out_channels,
            kernel_size=kernel_size,
            padding=int(kernel_size // 2),
            weight_attr=ParamAttr(initializer=weight_attr),
            bias_attr=False)
        self.eca_layerBlock = eca_layer(self.out_channels)

        self.shortcut = shortcut

    def forward(self, ins):
        x = self.in_conv(ins)
        if self.shortcut:
            out = x + self.eca_layerBlock(x)
        else:
            out = self.eca_layerBlock(x)
        return out

class DBECA(nn.Layer):
    def __init__(self, in_channels, out_channels, shortcut=True, **kwargs):
        super(RSEFPN, self).__init__()
        self.out_channels = out_channels
        self.ins_conv = nn.LayerList()
        self.inp_conv = nn.LayerList()

        for i in range(len(in_channels)):
            self.ins_conv.append(
                RECALayer(
                    in_channels[i],
                    out_channels,
                    kernel_size=1,
                    shortcut=shortcut))
            self.inp_conv.append(
                RECALayer(
                    out_channels,
                    out_channels // 4,
                    kernel_size=3,
                    shortcut=shortcut))

    def forward(self, x):
        c2, c3, c4, c5 = x

        in5 = self.ins_conv[3](c5)
        in4 = self.ins_conv[2](c4)
        in3 = self.ins_conv[1](c3)
        in2 = self.ins_conv[0](c2)

        out4 = in4 + F.upsample(
            in5, scale_factor=2, mode="nearest", align_mode=1)  # 1/16
        out3 = in3 + F.upsample(
            out4, scale_factor=2, mode="nearest", align_mode=1)  # 1/8
        out2 = in2 + F.upsample(
            out3, scale_factor=2, mode="nearest", align_mode=1)  # 1/4

        p5 = self.inp_conv[3](in5)
        p4 = self.inp_conv[2](out4)
        p3 = self.inp_conv[1](out3)
        p2 = self.inp_conv[0](out2)

        p5 = F.upsample(p5, scale_factor=8, mode="nearest", align_mode=1)
        p4 = F.upsample(p4, scale_factor=4, mode="nearest", align_mode=1)
        p3 = F.upsample(p3, scale_factor=2, mode="nearest", align_mode=1)

        fuse = paddle.concat([p5, p4, p3, p2], axis=1)
        return fuse
