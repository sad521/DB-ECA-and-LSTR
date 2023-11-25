# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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
from paddle.nn import Dropout, Linear, LayerNorm

from ppocr.modeling.heads.rec_ctc_head import get_para_bias_attr

class Mlp(nn.Layer):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
class ConvBNLayer(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 bias_attr=False,
                 groups=1,
                 act=nn.GELU):
        super().__init__()
        self.conv = nn.Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            weight_attr=paddle.ParamAttr(
                initializer=nn.initializer.KaimingUniform()),
            bias_attr=bias_attr)
        self.norm = nn.BatchNorm2D(out_channels)
        self.act = act()

    def forward(self, inputs):
        out = self.conv(inputs)
        out = self.norm(out)
        out = self.act(out)
        return out

class MultiheadAttention(nn.Layer):
    """Allows the model to jointly attend to information
    from different representation subspaces.
    See reference: Attention Is All You Need

    .. math::
        \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O
        \text{where} head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)

    Args:
        embed_dim: total dimension of the model
        num_heads: parallel attention layers, or heads

    """

    def __init__(self,
                 embed_dim,
                 num_heads,
                 dropout=0.,
                 bias=True,
                 add_bias_kv=False,
                 add_zero_attn=False):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim**-0.5
        self.out_proj = Linear(embed_dim, embed_dim, bias_attr=bias)
        self._reset_parameters()
        self.conv1 = paddle.nn.Conv2D(
            in_channels=embed_dim, out_channels=embed_dim, kernel_size=(1, 1))
        self.conv2 = paddle.nn.Conv2D(
            in_channels=embed_dim, out_channels=embed_dim, kernel_size=(1, 1))
        self.conv3 = paddle.nn.Conv2D(
            in_channels=embed_dim, out_channels=embed_dim, kernel_size=(1, 1))

    def _reset_parameters(self):
        xavier_uniform_(self.out_proj.weight)

    def forward(self,
                query,
                key,
                value,
                key_padding_mask=None,
                incremental_state=None,
                attn_mask=None):
        """
        Inputs of forward function
            query: [target length, batch size, embed dim]
            key: [sequence length, batch size, embed dim]
            value: [sequence length, batch size, embed dim]
            key_padding_mask: if True, mask padding based on batch size
            incremental_state: if provided, previous time steps are cashed
            need_weights: output attn_output_weights
            static_kv: key and value are static

        Outputs of forward function
            attn_output: [target length, batch size, embed dim]
            attn_output_weights: [batch size, target length, sequence length]
        """
        q_shape = paddle.shape(query)
        src_shape = paddle.shape(key)
        q = self._in_proj_q(query)
        k = self._in_proj_k(key)
        v = self._in_proj_v(value)
        q *= self.scaling
        q = paddle.transpose(
            paddle.reshape(
                q, [q_shape[0], q_shape[1], self.num_heads, self.head_dim]),
            [1, 2, 0, 3])
        k = paddle.transpose(
            paddle.reshape(
                k, [src_shape[0], q_shape[1], self.num_heads, self.head_dim]),
            [1, 2, 0, 3])
        v = paddle.transpose(
            paddle.reshape(
                v, [src_shape[0], q_shape[1], self.num_heads, self.head_dim]),
            [1, 2, 0, 3])
        if key_padding_mask is not None:
            assert key_padding_mask.shape[0] == q_shape[1]
            assert key_padding_mask.shape[1] == src_shape[0]
        attn_output_weights = paddle.matmul(q,
                                            paddle.transpose(k, [0, 1, 3, 2]))
        if attn_mask is not None:
            attn_mask = paddle.unsqueeze(paddle.unsqueeze(attn_mask, 0), 0)
            attn_output_weights += attn_mask
        if key_padding_mask is not None:
            attn_output_weights = paddle.reshape(
                attn_output_weights,
                [q_shape[1], self.num_heads, q_shape[0], src_shape[0]])
            key = paddle.unsqueeze(paddle.unsqueeze(key_padding_mask, 1), 2)
            key = paddle.cast(key, 'float32')
            y = paddle.full(
                shape=paddle.shape(key), dtype='float32', fill_value='-inf')
            y = paddle.where(key == 0., key, y)
            attn_output_weights += y
        attn_output_weights = F.softmax(
            attn_output_weights.astype('float32'),
            axis=-1,
            dtype=paddle.float32 if attn_output_weights.dtype == paddle.float16
            else attn_output_weights.dtype)
        attn_output_weights = F.dropout(
            attn_output_weights, p=self.dropout, training=self.training)

        attn_output = paddle.matmul(attn_output_weights, v)
        attn_output = paddle.reshape(
            paddle.transpose(attn_output, [2, 0, 1, 3]),
            [q_shape[0], q_shape[1], self.embed_dim])
        attn_output = self.out_proj(attn_output)

        return attn_output

    def _in_proj_q(self, query):
        query = paddle.transpose(query, [1, 2, 0])
        query = paddle.unsqueeze(query, axis=2)
        res = self.conv1(query)
        res = paddle.squeeze(res, axis=2)
        res = paddle.transpose(res, [2, 0, 1])
        return res

    def _in_proj_k(self, key):
        key = paddle.transpose(key, [1, 2, 0])
        key = paddle.unsqueeze(key, axis=2)
        res = self.conv2(key)
        res = paddle.squeeze(res, axis=2)
        res = paddle.transpose(res, [2, 0, 1])
        return res

    def _in_proj_v(self, value):
        value = paddle.transpose(value, [1, 2, 0])  #(1, 2, 0)
        value = paddle.unsqueeze(value, axis=2)
        res = self.conv3(value)
        res = paddle.squeeze(res, axis=2)
        res = paddle.transpose(res, [2, 0, 1])
        return res
class Attention(nn.Layer):
    def __init__(self,
                 dim,
                 num_heads=8,
                 mixer='Global',
                 HW=None,
                 local_k=[7, 11],
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias_attr=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.HW = HW
        if HW is not None:
            H = HW[0]
            W = HW[1]
            self.N = H * W
            self.C = dim
        if mixer == 'Local' and HW is not None:
            hk = local_k[0]
            wk = local_k[1]
            mask = paddle.ones([H * W, H + hk - 1, W + wk - 1], dtype='float32')
            for h in range(0, H):
                for w in range(0, W):
                    mask[h * W + w, h:h + hk, w:w + wk] = 0.
            mask_paddle = mask[:, hk // 2:H + hk // 2, wk // 2:W + wk //
                               2].flatten(1)
            mask_inf = paddle.full([H * W, H * W], '-inf', dtype='float32')
            mask = paddle.where(mask_paddle < 1, mask_paddle, mask_inf)
            self.mask = mask.unsqueeze([0, 1])
        self.mixer = mixer

    def forward(self, x):
        if self.HW is not None:
            N = self.N
            C = self.C
        else:
            _, N, C = x.shape
        qkv = self.qkv(x).reshape((0, N, 3, self.num_heads, C //
                                   self.num_heads)).transpose((2, 0, 3, 1, 4))
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]

        attn = (q.matmul(k.transpose((0, 1, 3, 2))))
        if self.mixer == 'Local':
            attn += self.mask
        attn = nn.functional.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x = (attn.matmul(v)).transpose((0, 2, 1, 3)).reshape((0, N, C))
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
class Block(nn.Layer):
    def __init__(self,
                 dim,
                 num_heads,
                 mixer='Global',
                 local_mixer=[7, 11],
                 HW=None,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer='nn.LayerNorm',
                 epsilon=1e-6,
                 prenorm=True):
        super().__init__()
        if isinstance(norm_layer, str):
            self.norm1 = eval(norm_layer)(dim, epsilon=epsilon)
        else:
            self.norm1 = norm_layer(dim)
        if mixer == 'Global' or mixer == 'Local':
            self.mixer = Attention(
                dim,
                num_heads=num_heads,
                mixer=mixer,
                HW=HW,
                local_k=local_mixer,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop)
        elif mixer == 'Conv':
            self.mixer = ConvMixer(
                dim, num_heads=num_heads, HW=HW, local_k=local_mixer)
        else:
            raise TypeError("The mixer must be one of [Global, Local, Conv]")

        self.drop_path = DropPath(drop_path) if drop_path > 0. else Identity()
        if isinstance(norm_layer, str):
            self.norm2 = eval(norm_layer)(dim, epsilon=epsilon)
        else:
            self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp_ratio = mlp_ratio
        self.mlp = Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer,
                       drop=drop)
        self.prenorm = prenorm

    def forward(self, x):
        if self.prenorm:
            x = self.norm1(x + self.drop_path(self.mixer(x)))
            x = self.norm2(x + self.drop_path(self.mlp(x)))
        else:
            x = x + self.drop_path(self.mixer(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x



class InferenceBlock(nn.Layer):
    def __init__(self, in_channels,
                 hidden_size,
                 d_model=512,
                 nhead=8,
                 dim_feedforward=1024,
                 attention_dropout_rate=0.0,
                 with_self_attn=True,
                 with_cross_attn=False,
                 epsilon=1e-5,
                 residual_dropout_rate=0.1
                 ):
        super(InferenceBlock, self).__init__()
        self.with_self_attn = with_self_attn
        self.out_channels = hidden_size * 2
        #self.lstm = nn.LSTM(
           # in_channels, hidden_size, direction='bidirectional', num_layers=2)
        self.self_attn = MultiheadAttention(
            d_model,
            nhead,
            dropout=attention_dropout_rate,
            self_attn=with_self_attn)
        self.norm1 = LayerNorm(d_model, epsilon=epsilon)
        self.dropout1 = Dropout(residual_dropout_rate)
        self.with_cross_attn = with_cross_attn
        if with_cross_attn:
            self.cross_attn = MultiheadAttention(  # for self_attn of encoder or cross_attn of decoder
                d_model,
                nhead,
                dropout=attention_dropout_rate)
            self.norm2 = LayerNorm(d_model, epsilon=epsilon)
            self.dropout2 = Dropout(residual_dropout_rate)
        self.mlp = Mlp(in_features=d_model,
                       hidden_features=dim_feedforward,
                       act_layer=nn.ReLU,
                       drop=residual_dropout_rate)

        self.norm3 = LayerNorm(d_model, epsilon=epsilon)
        self.dropout3 = Dropout(residual_dropout_rate)

    def forward(self, x, memory=None, self_mask=None, cross_mask=None):

        # x, _ = self.lstm(x)
        if self.with_self_attn:
            x1 = self.self_attn(x, attn_mask=self_mask)
            x2 = x + self.dropout1(x1)
        x = self.norm3(x2 + self.mlp(self.norm1(x2)))
        return x


class EncoderWithFC(nn.Layer):
    def __init__(self, in_channels, hidden_size):
        super(EncoderWithFC, self).__init__()
        self.out_channels = hidden_size
        weight_attr, bias_attr = get_para_bias_attr(
            l2_decay=0.00001, k=in_channels)
        self.fc = nn.Linear(
            in_channels,
            hidden_size,
            weight_attr=weight_attr,
            bias_attr=bias_attr,
            name='reduce_encoder_fea')

    def forward(self, x):
        x = self.fc(x)
        return x


class SequenceEncoder(nn.Layer):
    def __init__(self, in_channels, encoder_type, hidden_size=48, **kwargs):
        super(SequenceEncoder, self).__init__()
        self.encoder_reshape = Im2Seq(in_channels)
        self.out_channels = self.encoder_reshape.out_channels
        if encoder_type == 'reshape':
            self.only_reshape = True
        else:
            support_encoder_dict = {
                'reshape': Im2Seq,
                'fc': EncoderWithFC,
                'rnn': InferenceBlock
            }
            assert encoder_type in support_encoder_dict, '{} must in {}'.format(
                encoder_type, support_encoder_dict.keys())

            self.encoder = support_encoder_dict[encoder_type](
                self.encoder_reshape.out_channels, hidden_size)
            self.out_channels = self.encoder.out_channels
            self.only_reshape = False

    def forward(self, x):
        x = self.encoder_reshape(x)
        if not self.only_reshape:
            x = self.encoder(x)
        return x
