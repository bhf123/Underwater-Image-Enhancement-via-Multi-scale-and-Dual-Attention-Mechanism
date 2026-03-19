import torch
import torch.nn as nn
import torch.nn.functional as F
from var_pool import VarPool2d

# 拉普拉斯卷积核，用于边缘检测
class Laplace(nn.Module):
    def __init__(self):
        super(Laplace, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, stride=1, padding=0, bias=False)
        nn.init.constant_(self.conv1.weight, 1)
        nn.init.constant_(self.conv1.weight[0, 0, 1, 1], -8)
        nn.init.constant_(self.conv1.weight[0, 1, 1, 1], -8)
        nn.init.constant_(self.conv1.weight[0, 2, 1, 1], -8)

    def forward(self, x):
        edge_map = self.conv1(x)
        return edge_map

class PALayer(nn.Module):
    '''空间注意力，输入时乘以通道注意力权重，实现交叉融合'''
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.var_pool = VarPool2d()
        self.conv = nn.Conv2d(channel * 3, channel, kernel_size=1, padding=0, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, ca_weight):
        # x: 输入特征 (B,C,H,W)
        # ca_weight: 通道权重 (B,C,1,1)，用于条件调节空间权重
        x_cond = x * ca_weight  # 按通道广播乘
        max_pool = self.max_pool(x_cond)
        var_pool = self.var_pool(x_cond)
        avg_pool = self.avg_pool(x_cond)
        pooled = torch.cat([max_pool, var_pool, avg_pool], dim=1)
        y = self.conv(pooled)
        y = self.sigmoid(y)  # 输出空间注意力权重，尺寸 (B,C,1,1)
        return y

class CALayer(nn.Module):
    '''通道注意力，输入时乘以空间注意力权重，实现交叉融合'''
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.CA = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x, pa_weight):
        # x: 输入特征 (B,C,H,W)
        # pa_weight: 空间权重 (B,C,H,W) 或 (B,C,1,1)
        # 先做全局平均池化获得通道权重条件
        if pa_weight.shape[2] != 1 or pa_weight.shape[3] != 1:
            pa_weight_pool = pa_weight.mean(dim=(2, 3), keepdim=True)  # (B,C,1,1)
        else:
            pa_weight_pool = pa_weight
        x_cond = x * pa_weight_pool
        y = self.avg_pool(x_cond)
        y = self.CA(y)
        return y  # 通道权重 (B,C,1,1)

class Block(nn.Module):
    '''带交叉注意力的残差块'''
    def __init__(self, dim, kernel_size):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size, padding=kernel_size // 2, bias=True)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size, padding=kernel_size // 2, bias=True)
        self.calayer = CALayer(dim)
        self.palayer = PALayer(dim)

    def forward(self, x):
        res = self.conv1(x)
        # 初始化空间注意力权重全1
        pa_weight = torch.ones_like(res)
        # 交叉迭代计算权重（可迭代多次，这里两次示范）
        ca_weight = self.calayer(res, pa_weight)
        pa_weight = self.palayer(res, ca_weight)
        ca_weight = self.calayer(res, pa_weight)
        # 融合加权
        res_ca = res * ca_weight
        res_pa = res * pa_weight
        res = res_ca + res_pa
        res = self.conv2(res)
        res = res + x
        return res

class GS(nn.Module):
    '''复合分组模块'''
    def __init__(self, dim, kernel_size, blocks):
        super(GS, self).__init__()
        self.gs = nn.Sequential(*[Block(dim, kernel_size) for _ in range(blocks)])

    def forward(self, x):
        return self.gs(x)

class Branch(nn.Module):
    '''分支单元模块'''
    def __init__(self, in_channels, out_channels, kernel_size, bias=True):
        super(Branch, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=bias)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=bias)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.IN = nn.InstanceNorm2d(out_channels)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.relu(x1)
        x1 = self.IN(x1)
        x2 = self.conv2(x1)
        x3 = self.relu(x2)
        x4 = self.IN(x3)
        x5 = self.conv2(x4)
        return x1, x5

class LANet(nn.Module):
    def __init__(self, gps=3, blocks=20, dim=64, kernel_size=3):
        super(LANet, self).__init__()
        self.gps = gps
        self.dim = dim
        self.kernel_size = kernel_size
        self.laplace = Laplace()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        assert self.gps == 3
        self.g1 = GS(self.dim, kernel_size, blocks=blocks)
        self.g2 = GS(self.dim, kernel_size, blocks=blocks)
        self.g3 = GS(self.dim, kernel_size, blocks=blocks)
        self.branch_3 = Branch(in_channels=3, out_channels=self.dim, kernel_size=1)
        self.branch_5 = Branch(in_channels=3, out_channels=self.dim, kernel_size=3)
        self.branch_7 = Branch(in_channels=3, out_channels=self.dim, kernel_size=5)
        self.fusion = nn.Sequential(
            nn.Conv2d(self.dim * self.gps, self.dim // 8, 3, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.InstanceNorm2d(self.dim // 8),
            nn.Conv2d(self.dim // 8, self.gps, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
        self.Final = nn.Conv2d(self.dim, 3, 1, padding=0, bias=True)

    def forward(self, x):
        x11, x1 = self.branch_3(x)
        x22, x2 = self.branch_5(x)
        x33, x3 = self.branch_7(x)

        w = self.fusion(torch.cat([x1, x2, x3], dim=1))
        w = torch.split(w, 1, dim=1)
        x4 = w[0] * x1 + w[1] * x2 + w[2] * x3

        res1 = self.g1(x4)
        x5 = self.avg_pool(x4)
        res1 = x5 * res1 + x33

        res2 = self.g2(res1)
        x6 = self.avg_pool(res1)
        res2 = x6 * res2 + x22

        res3 = self.g3(res2)
        x7 = self.avg_pool(res2)
        res3 = x7 * res3 + x11

        out = self.Final(res3)

        edge_map = self.laplace(out)

        return out, edge_map
