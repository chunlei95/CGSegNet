import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import to_2tuple


# noinspection PyTypeChecker
class CGSegNet(nn.Module):
    """输入图像大小设置为256 x 256，16个patch，每个patch的大小为16 x 16

    :param in_dim: transformer结构的输入维度，也即PatchEmbedding中的embed_dim
    :param out_dim: 整个transformer编码器的最终输出的维度
    """

    def __init__(self, image_channel=1, base_channel=64, num_class=1, in_dim=1024, out_dim=1024, heads=4,
                 num_blocks=5):
        super(CGSegNet, self).__init__()
        self.cnn_encoder = CNNEncoder(in_channel=image_channel, base_channel=base_channel, num_blocks=num_blocks)
        self.transformer_encoder = TransformerEncoder(in_dim, out_dim, heads, num_blocks=num_blocks)
        self.gene_decoder = GeneDecoder(in_channels=base_channel * (2 ** (num_blocks - 1)), num_class=num_class,
                                        num_blocks=num_blocks)
        # 目前考虑全监督方式，暂时不需要下面的映射头
        # self.global_projection_head = GlobalProjectionHead()
        # self.dense_projection_head = DenseProjectionHead()
        self.fusion = CTFusion()
        self.conv1x1 = nn.Conv2d(base_channel, num_class, kernel_size=1)
        self.patch_embedding = PatchEmbedding()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        cnn_results = self.cnn_encoder(x)
        transformer_input = self.patch_embedding(x)
        transformer_results = self.transformer_encoder(transformer_input)
        fusion_results = self.fusion(cnn_results, transformer_results)
        decoder_result = self.gene_decoder(fusion_results[:len(fusion_results) - 1],
                                           fusion_results[len(fusion_results) - 1])
        out = self.sigmoid(self.conv1x1(decoder_result))
        return out


# class TransformerBlock(nn.Module):
#
#     def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
#                  proj_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
#         super(TransformerBlock, self).__init__()
#         # 有的论文中没有用layer norm，用的是batch norm，layer norm一般是用在自然语言处理中，
#         # batch norm一般用在计算机视觉中，这个地方需要根据实验结果进行调节
#         self.norm1 = norm_layer(dim)
#         self.attn = MultiHeadAttention(
#             dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=proj_drop)
#         # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         self.norm2 = norm_layer(dim)
#         mlp_hidden_dim = int(dim * mlp_ratio)
#         self.mlp = FFN(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
#         # todo 这段代码中没有考虑位置编码
#
#     def forward(self, x):
#         x = x + self.drop_path(self.attn(self.norm1(x)))
#         x = x + self.drop_path(self.mlp(self.norm2(x)))
#         return x


class TransformerEncoder(nn.Module):
    """transformer encoder，不进行类似下采样的操作，即特征图的大小不变，但是中间层的维度是变化的（为了方便与CNN编码器的特征进行融合）

    :param in_dims: 输入图像patch序列的维度，即图像patch的个数
    :param out_dims:
    :param heads: 多头注意力模块头的数量
    :param num_blocks: transformer层的个数，一个transformer层包括一个多头注意力模块和一个mlp模块，与CNN编码器的层数相同
    :param attn_drop: softmax(..)后的dropout层的dropout参数，默认为0.
    :param proj_drop: to_out后的dropout层的dropout参数，默认为0.
    :param sub_sample: 是否需要对k, v进行下采样，默认不进行下采样
    :param reduce_size: 如果下采样，下采样后的尺寸，默认为16 x 16
    :param projection: 进行下采样的方式，'interp'或'maxpool'，默认为'interp'
    :param rel_pos: 是否需要添加相对位置编码，默认为True
    """

    def __init__(self, in_dims, out_dims, heads=4, num_blocks=4, attn_drop=0., proj_drop=0., sub_sample=False,
                 reduce_size=16, projection='interp', rel_pos=True):
        super(TransformerEncoder, self).__init__()
        # ***************** 设置每一个transformer层的输入维度 **************************************************************
        # 如何处理多层transformer之间的维度变化，并使其能与CNN分支进行融合需要很巧妙的设计
        self.in_dims = []
        for i in range(num_blocks):
            # self.in_dims.append(in_dims // (2 ** i))
            self.in_dims.append(in_dims)
        # **************** 设置每一个transformer层的输出维度 ***************************************************************
        self.out_dims = []
        for i in range(num_blocks):
            # self.out_dims.append(out_dims * (2 ** (num_blocks - i - 1)))
            self.out_dims.append(out_dims)
        # **************************************************************************************************************
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            self.blocks.append(
                TransformerBlock(self.in_dims[i], self.out_dims[i], heads, sub_sample, attn_drop, proj_drop,
                                 reduce_size, projection, rel_pos))

    def forward(self, x):
        layer_results = []
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            layer_results.append(self.blocks[i](x))
        # todo 每一层的输出结果都需要转换成2D形式(在fusion模块中进行，也可以在此处进行,可能需要用到每一层的dim值，因此在此处进行可能比较合适)
        # 暂时不需要处理上面的todo
        return layer_results


# noinspection PyTypeChecker
class TransformerBlock(nn.Module):
    """代码借鉴：https://github.com/yhygao/UTNet
        transformer块
    """

    def __init__(self, in_planes, out_planes, heads, sub_sample=False, attn_drop=0., proj_drop=0., reduce_size=16,
                 projection='interp',
                 rel_pos=True):
        super(TransformerBlock, self).__init__()

        # **************************************************************************************************************
        # todo 使用layer norm还是batch norm要看实验结果
        self.bn1 = nn.BatchNorm2d(in_planes)
        # **************************************************************************************************************
        self.attn = MultiHeadAttention(in_planes, out_planes, heads=heads, dim_head=in_planes // heads,
                                       sub_sample=sub_sample, attn_drop=attn_drop, proj_drop=proj_drop,
                                       reduce_size=reduce_size, projection=projection, rel_pos=rel_pos)

        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.mlp = nn.Conv2d(in_planes, in_planes, kernel_size=1, bias=False)
        # conv1x1 has no difference with mlp in performance
        self.shortcut = nn.Sequential()
        if in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)
            )

    def forward(self, x):
        out = self.bn1(x)
        out, q_k_attn = self.attn(out)

        out = out + self.shortcut(x)
        residue = out

        out = self.bn2(out)
        out = self.relu(out)
        out = self.mlp(out)

        out += residue

        return out


# noinspection PyPep8Naming,SpellCheckingInspection
class MultiHeadAttention(nn.Module):
    """代码借鉴：https://github.com/yhygao/UTNet
        多头注意力模块
    """

    def __init__(self, in_dim, out_dim, heads, dim_head, sub_sample=False, reduce_size=16, projection='interp',
                 attn_drop=0.,
                 proj_drop=0., rel_pos=True):
        super(MultiHeadAttention, self).__init__()
        self.inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** (-0.5)
        self.dim_head = dim_head
        self.reduce_size = reduce_size
        self.projection = projection
        self.rel_pos = rel_pos

        # depth-wise conv is slightly better than conv1x1
        # self.to_qkv = nn.Conv2d(dim, self.inner_dim*3, kernel_size=1, stride=1, padding=0, bias=True)
        # self.to_out = nn.Conv2d(self.inner_dim, dim, kernel_size=1, stride=1, padding=0, bias=True)

        self.to_qkv = depthwise_separable_conv(in_dim, self.inner_dim * 3)
        self.to_out = depthwise_separable_conv(self.inner_dim, out_dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.sub_sample = sub_sample

        if self.rel_pos:
            # 2D input-independent relative position encoding is a little better than
            # 1D input-dependent counterpart
            self.relative_position_encoding = RelativePositionEncoding(heads, reduce_size, reduce_size)
            # self.relative_position_encoding = RelativePositionEmbedding(dim_head, reduce_size)

    def forward(self, x):
        B, C, H, W = x.shape

        # B, inner_dim, H, W
        qkv = self.to_qkv(x)
        q, k, v = qkv.chunk(3, dim=1)

        # ********************** 对k，v进行下采样的操作不一定用 *************************************************************
        if self.sub_sample:
            if self.projection == 'interp' and H != self.reduce_size:
                k, v = map(lambda t: F.interpolate(t, size=self.reduce_size, mode='bilinear', align_corners=True),
                           (k, v))

            elif self.projection == 'maxpool' and H != self.reduce_size:
                k, v = map(lambda t: F.adaptive_max_pool2d(t, output_size=self.reduce_size), (k, v))
        # **************************************************************************************************************

        q = rearrange(q, 'b (dim_head heads) h w -> b heads (h w) dim_head', dim_head=self.dim_head, heads=self.heads,
                      h=H, w=W)
        k, v = map(lambda t: rearrange(t, 'b (dim_head heads) h w -> b heads (h w) dim_head', dim_head=self.dim_head,
                                       heads=self.heads, h=self.reduce_size, w=self.reduce_size), (k, v))

        q_k_attn = torch.einsum('bhid,bhjd->bhij', q, k)

        if self.rel_pos:
            relative_position_bias = self.relative_position_encoding(H, W)
            q_k_attn += relative_position_bias
            # rel_attn_h, rel_attn_w = self.relative_position_encoding(q, self.heads, H, W, self.dim_head)
            # q_k_attn = q_k_attn + rel_attn_h + rel_attn_w

        q_k_attn *= self.scale
        q_k_attn = F.softmax(q_k_attn, dim=-1)
        q_k_attn = self.attn_drop(q_k_attn)

        out = torch.einsum('bhij,bhjd->bhid', q_k_attn, v)
        out = rearrange(out, 'b heads (h w) dim_head -> b (dim_head heads) h w', h=H, w=W, dim_head=self.dim_head,
                        heads=self.heads)

        out = self.to_out(out)
        out = self.proj_drop(out)

        return out, q_k_attn


# noinspection PyPep8Naming,SpellCheckingInspection,PyTypeChecker
class depthwise_separable_conv(nn.Module):
    """代码借鉴：https://github.com/yhygao/UTNet
        深度可分离卷积
    """

    def __init__(self, in_ch, out_ch, stride=1, kernel_size=3, padding=1, bias=False):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size, padding=padding, groups=in_ch, bias=bias,
                                   stride=stride)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)

        return out


class PatchEmbedding(nn.Module):
    """ Image to Patch Embedding

    只需要在初始输入时执行一次
    代码借鉴：https://github.com/Rayicer/TransFuse
    源代码中img_size=224, in_channel=3, embed_dim=768
    """

    def __init__(self, img_size=256, patch_size=16, in_channel=1, embed_dim=1024):
        super(PatchEmbedding, self).__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_channel, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape

        # FIXME look at relaxing size constraints
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #    f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


# noinspection PyPep8Naming,SpellCheckingInspection
class RelativePositionEncoding(nn.Module):
    """代码借鉴：https://github.com/yhygao/UTNet
        相对位置编码
    """

    def __init__(self, num_heads, h, w):
        super(RelativePositionEncoding, self).__init__()
        self.num_heads = num_heads
        self.h = h
        self.w = w

        self.relative_position_bias_table = nn.Parameter(
            torch.randn((2 * h - 1) * (2 * w - 1), num_heads) * 0.02)

        coords_h = torch.arange(self.h)
        coords_w = torch.arange(self.w)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, h, w
        coords_flatten = torch.flatten(coords, 1)  # 2, hw

        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.h - 1
        relative_coords[:, :, 1] += self.w - 1
        relative_coords[:, :, 0] *= 2 * self.h - 1
        relative_position_index = relative_coords.sum(-1)  # hw, hw

        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self, H, W):
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(self.h,
                                                                                                               self.w,
                                                                                                               self.h * self.w,
                                                                                                               -1)  # h, w, hw, nH
        relative_position_bias_expand_h = torch.repeat_interleave(relative_position_bias, H // self.h, dim=0)
        relative_position_bias_expanded = torch.repeat_interleave(relative_position_bias_expand_h, W // self.w,
                                                                  dim=1)  # HW, hw, nH

        relative_position_bias_expanded = relative_position_bias_expanded.view(H * W, self.h * self.w,
                                                                               self.num_heads).permute(2, 0,
                                                                                                       1).contiguous().unsqueeze(
            0)

        return relative_position_bias_expanded


class FFN(nn.Module):
    """Feed forward network, 紧接在multi-head attention后面

    代码借鉴：https://github.com/Rayicer/TransFuse
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super(FFN, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        # 注意x的形状不是2d的，而是（B，N，dim）
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# noinspection PyTypeChecker
class CNNEncoder(nn.Module):
    """ 以CNN构建的编码器
    todo 此处代码需要进行大改，不然无法进行CNN和Transformer特征的多尺度融合(已修改)

    :param in_channel: 图像的初始通道
    :param base_channel: 图像在模型重的基本通道（其它通道数都是该通道的倍数）
    :param num_blocks: CNN编码器重双卷积块（U-Net模型中的）的个数，与Transformer编码器中transformer的层数相同
    """

    def __init__(self, in_channel, base_channel=64, num_blocks=5):
        super(CNNEncoder, self).__init__()
        # ******** 修改输入图像的通道到基础值 *******************************************************************************
        # self.change_channel = nn.Conv2d(in_channel, base_channel, kernel_size=1)
        # self.bn = nn.BatchNorm2d(base_channel)
        # self.relu = nn.ReLU(inplace=True)
        # ******** 设置每一层的通道数 *************************************************************************************
        channels = []
        for i in range(num_blocks):
            channels.append(base_channel * (2 ** i))
        # ******** 构建transformer编码器（多个transformer块的堆叠）**********************************************************
        print(channels)
        self.blocks = nn.ModuleList(
            [BasicEncoderBlock(in_channel, channels[0], channels[0], down_sample=False, kernel_size=2)])
        for i in range(num_blocks - 1):
            self.blocks.append(BasicEncoderBlock(channels[i], channels[i + 1], channels[i + 1]))

    def forward(self, x):
        layer_result = []
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            layer_result.append(x)
        return layer_result


class BasicEncoderBlock(nn.Module):
    """
    :param down_sample: 是否需要进行下采样
    :param kernel_size: 最大池化层使用的kernel_size
    :param bottleneck: 使用基本的残差块还是使用bottleneck block，默认使用基本的残差块
    """

    def __init__(self, in_channel, mid_channel, out_channel, down_sample=True, kernel_size=2, bottleneck=False):
        super(BasicEncoderBlock, self).__init__()
        blocks = []
        if down_sample:
            blocks.append(nn.MaxPool2d(kernel_size=kernel_size))
        if bottleneck:
            blocks.append(BottleneckBlock(in_channel, mid_channel))
            blocks.append(BottleneckBlock(mid_channel, out_channel))
        else:
            blocks.append(BasicBlock(in_channel, mid_channel))
            blocks.append(BasicBlock(mid_channel, out_channel))
        self.layers = nn.Sequential(*blocks)

    def forward(self, x):
        return self.layers(x)


# noinspection PyTypeChecker
class CTFusion(nn.Module):
    """CNN和transformer融合模块

    """

    def __init__(self):
        super(CTFusion, self).__init__()
        self.t_blocks = nn.ModuleList([
            nn.Conv2d(4, 64, kernel_size=1, bias=False),
            nn.Conv2d(16, 128, kernel_size=1, bias=False),
            nn.Conv2d(64, 256, kernel_size=1, bias=False),
            nn.Conv2d(256, 512, kernel_size=1, bias=False)
        ])

        self.pool = nn.MaxPool2d(kernel_size=2)

        self.c_blocks = nn.ModuleList([
            nn.Conv2d(64, 128, kernel_size=1, bias=False),
            nn.Conv2d(128, 256, kernel_size=1, bias=False),
            nn.Conv2d(256, 512, kernel_size=1, bias=False),
            nn.Conv2d(512, 1024, kernel_size=1, bias=False)
        ])

    def forward(self, cnn_layer_results, transformer_layer_results):
        assert len(cnn_layer_results) == len(transformer_layer_results)
        length = len(cnn_layer_results)
        fusion_result = []
        for i in range(length):
            t_i = transformer_layer_results[i].view(cnn_layer_results[i].shape[0], -1, cnn_layer_results[i].shape[-2],
                                                    cnn_layer_results[i].shape[-1])
            if t_i.shape[1] != cnn_layer_results[i].shape[1]:
                t_i = self.t_blocks[i](t_i)
            c_i = torch.zeros_like(t_i)
            if i != 0:
                c_i = self.pool(self.c_blocks[i - 1](fusion_result[i - 1]))
            f_i = t_i + c_i + cnn_layer_results[i]
            fusion_result.append(f_i)
        return fusion_result


# noinspection PyTypeChecker
class GeneDecoder(nn.Module):
    """生成分支的解码器

    """

    def __init__(self, in_channels, num_class=1, num_blocks=5):
        super(GeneDecoder, self).__init__()
        self.num_class = num_class
        channels = []
        for i in range(num_blocks):
            mid_channels = in_channels // (2 ** i)
            assert mid_channels > 0
            channels.append(mid_channels)

        print(in_channels)
        print(channels)

        blocks = []
        for i in range(num_blocks - 1):
            blocks.append(
                BasicDecoderBlock(channels[i], channels[i + 1], channels[i + 1], bottleneck=False, up_sample=True,
                                  factor=2))
        # blocks.append(nn.Conv2d(channels[num_blocks - 1], num_class, kernel_size=1))
        # 注意，此处的分割结果没有进行bn + sigmoid/softmax ******************************************************************
        self.layers = nn.Sequential(*blocks)

    def forward(self, encoder_results, x):
        for i in range(len(encoder_results)):
            x = self.layers[i](encoder_results[len(encoder_results) - 1 - i], x)
        return x


# noinspection PyTypeChecker
class BasicDecoderBlock(nn.Module):

    def __init__(self, in_channel, mid_channel, out_channel, bottleneck=False, up_sample=True, factor=2):
        super(BasicDecoderBlock, self).__init__()
        self.up_sample = None
        if up_sample:
            self.up_sample = nn.ConvTranspose2d(in_channel, mid_channel, stride=factor, kernel_size=factor)
        blocks = []
        if bottleneck:
            blocks.append(BottleneckBlock(in_channel, mid_channel))
            blocks.append(BottleneckBlock(mid_channel, out_channel))
        else:
            blocks.append(BasicBlock(in_channel, mid_channel))
            blocks.append(BasicBlock(mid_channel, out_channel))
        self.layers = nn.Sequential(*blocks)

    def forward(self, x1, x2):
        # x1来自fusion模块，x2来自上一层的特征图，如果x1的形状与x2不相同，将x1转换成x2的形状
        # x1和x2的形状均为(N, C, H, W)
        if self.up_sample is not None:
            x2 = self.up_sample(x2)
        if x1.shape[-1] != x2.shape[-1]:
            # 对x1进行插值处理，使其与x2的形状相同
            x1 = F.interpolate(x1, x2.shape, mode='bilinear', align_corners=True)
        # 将x1与x2在通道维进行连接
        # todo 此处需要注意维度的变化
        x = torch.cat([x1, x2], dim=1)
        return self.layers(x)


# noinspection PyTypeChecker
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


# noinspection PyTypeChecker
def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)


# noinspection PyTypeChecker
class BasicBlock(nn.Module):
    """代码借鉴：https://github.com/yhygao/UTNet
        普通残差块
    """

    def __init__(self, in_planes, out_planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        residue = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)

        out += self.shortcut(residue)

        out = self.bn2(out)
        out = self.relu(out)
        return out


# noinspection PyTypeChecker
class BottleneckBlock(nn.Module):
    """残差网络中的bottleneck block

    """

    def __init__(self, in_channel, out_channel, stride=1):
        super(BottleneckBlock, self).__init__()
        # **************************************************************************************************************
        mid_channel = in_channel // 4
        # **************************************************************************************************************
        self.conv1 = nn.Conv2d(in_channel, mid_channel, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(mid_channel)
        self.conv2 = nn.Conv2d(mid_channel, mid_channel, kernel_size=3, stride=stride)
        self.bn2 = nn.BatchNorm2d(mid_channel)
        self.conv3 = nn.Conv2d(mid_channel, out_channel, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(out_channel)
        # **************************************************************************************************************
        self.shortcut = nn.Sequential()
        if in_channel != out_channel or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1),
            )
        # **************************************************************************************************************

    def forward(self, x):
        residual = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.conv3(x)

        x += self.shortcut(residual)
        x = self.relu(self.bn3(x))
        return x


class GlobalProjectionHead(nn.Module):
    """全局映射头

    """

    def __init__(self):
        super(GlobalProjectionHead, self).__init__()

    def forward(self, x):
        pass


class DenseProjectionHead(nn.Module):
    """密集映射头
    """

    def __init__(self):
        super(DenseProjectionHead, self).__init__()

    def forward(self, x):
        pass
