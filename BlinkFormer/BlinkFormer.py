import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)  # 正则化
        self.fn = fn  # 具体的操作

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        # 前向传播
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    # attention
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads  # 计算最终进行全连接操作时输入神经元的个数
        project_out = not (heads == 1 and dim_head == dim)  # 多头注意力并且输入和输出维度相同时为True

        self.heads = heads  # 多头注意力中“头”的个数
        self.scale = dim_head ** -0.5  # 缩放操作，论文 Attention is all you need 中有介绍

        self.attend = nn.Softmax(dim=-1)  # 初始化一个Softmax操作
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)  # 对Q、K、V三组向量先进性线性操作

        # 线性全连接，如果不是多头或者输入输出维度不相等，进行空操作
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads  # 获得输入x的维度和多头注意力的“头”数
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # 先对Q、K、V进行线性操作，然后chunk乘三三份
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)  # 整理维度，获得Q、K、V

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale  # Q, K 向量先做点乘，来计算相关性，然后除以缩放因子

        attn = self.attend(dots)  # 做Softmax运算

        out = einsum('b h i j, b h j d -> b h i d', attn, v)  # Softmax运算结果与Value向量相乘，得到最终结果
        out = rearrange(out, 'b h n d -> b n (h d)')  # 重新整理维度
        return self.to_out(out)  # 做线性的全连接操作或者空操作（空操作直接输出out）


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])  # Transformer包含多个编码器的叠加
        for _ in range(depth):
            # 编码器包含两大块：自注意力模块和前向传播模块
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),  # 多头自注意力模块
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))  # 前向传播模块
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            # 自注意力模块和前向传播模块都使用了残差的模式
            x = attn(x) + x
            x = ff(x) + x
        return x


class BlinkFormer(nn.Module):
    def __init__(self, seq_lenth=13, num_classes=2, dim=1024, depth=6, heads=16, mlp_dim=2048,  channels=3, dim_head=64, dropout=0.1, emb_dropout=0.):
        super().__init__()
        self.image_size = 48
        patch_dim = channels * self.image_size ** 2  # 线性变换时的输入大小，即每一个图像宽、高、通道的乘积
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b seq c h w -> b seq (c h w)'),
            # 将批量为b通道为c高为h*p1宽为w*p2的图像转化为批量为b个数为h*w维度为p1*p2*c的图像块
            # 即，把b张c通道的图像分割成b*（h*w）张大小为P1*p2*c的图像块
            # 例如：patch_size为16  (8, 3, 48, 48)->(8, 9, 768)
            nn.Linear(patch_dim, dim),  # 对分割好的图像块进行线性处理（全连接），输入维度为每一个小块的所有像素个数，输出为dim（函数传入的参数）
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, seq_lenth + 1, dim))  # 位置编码，获取一组正态分布的数据用于训练
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))  # 分类令牌，可训练
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)  # Transformer模块

        self.to_latent = nn.Identity()  # 占位操作

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),  # 正则化
            nn.Linear(dim, num_classes)  # 线性输出
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)  # 切块操作，shape (b, n, dim)，b为批量，n为切块数目，dim为最终线性操作时输入的神经元个数
        b, n, _ = x.shape  # shape (b, n, 1024)

        cls_tokens = repeat(self.cls_token, '() n d -> b n d',
                            b=b)  # 分类令牌，将self.cls_token（形状为1, 1, dim）赋值为shape (b, 1, dim)
        x = torch.cat((cls_tokens, x), dim=1)  # 将分类令牌拼接到输入中，x的shape (b, n+1, 1024)
        x += self.pos_embedding[:, :(n + 1)]  # 进行位置编码，shape (b, n+1, 1024)
        x = self.dropout(x)

        x = self.transformer(x)  # transformer操作

        x = x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)  # 线性输出
    


class BlinkFormer_with_BSE_head(nn.Module):
    def __init__(self, seq_length=13, num_classes=2, dim=1024, depth=6, heads=16, mlp_dim=2048, channels=3, dim_head=64, dropout=0.1, emb_dropout=0.):
        super().__init__()
        self.image_size = 48
        patch_dim = channels * self.image_size ** 2
       
        # MLP头1用于二分类任务
        self.cls_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

        # MLP头2用于眨眼强度预测，包含13个输出节点
        self.bse_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 256),
            nn.Linear(256, 128),
            nn.Linear(128, 1)
        )

        # Softmax运算应用在MLP头2的输出上
        self.softmax = nn.Softmax(dim=-1)

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b seq c h w -> b seq (c h w)'),
            nn.Linear(patch_dim, dim)
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_length + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(
            dim, depth, heads, dim_head, mlp_dim, dropout)

        self.to_latent = nn.Identity()
        self.seq_length = seq_length

        self.sigmoid = nn.Sigmoid()

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, d = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        # print(x.shape)
        x1 = x[:, 0]
        x2 = x[:, 1:]
        # print(x1.shape, x2.shape)

        # MLP头1用于二分类任务，输入为x1
        out1 = self.cls_head(x1)

        # MLP头2用于眨眼强度预测，包含13个输出节点，输入为x
        out2 = self.bse_head(x2)
        out2 = out2.squeeze()
        out2 = self.sigmoid(out2)
        return out1, out2


if __name__ == '__main__':
    model = BlinkFormer(num_classes=2)
    x = torch.randn([32, 13, 3, 48, 48])
    cls, reg = model(x)
    print(cls.shape)
    print(reg.shape)
