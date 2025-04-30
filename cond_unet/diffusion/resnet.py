import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange
from typing import Type

# self define funtion also same as the function in the denoising_diffusion_pytorch.py
def exists(x):
    return x is not None

# here we choose the group norm instead of RMSnorm used in the origin codes
# resnet reference: diffusion forcing
class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

# used for the u net backcone
class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, emb_dim=None, groups=8):
        """
        :param dim: input channel
        :param dim_out:  output channel
        :param emb_dim: extra embedding to fuse, such as time or control
        :param groups: group for conv2d
        """
        super().__init__()
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, dim_out * 2)) if exists(emb_dim) else None

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, emb=None):
        scale_shift = None
        if exists(self.mlp) and exists(emb):
            emb = self.mlp(emb)
            emb = rearrange(emb, "b c -> b c 1 1")
            scale_shift = emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)

# here is for process the condition information
class ResBlock2d(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, dim=32, size=64, stride=1, activation: Type[nn.Module] = nn.ReLU):
        super(ResBlock2d, self).__init__()
        """
        dim: linear output;
        size: the image size
        """
        self.activation = activation()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.shortcut = nn.Identity()
        self.linear = nn.Linear(planes*size**2, dim)
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
            )

    def forward(self, x):
        out = self.activation(self.conv1(x))
        out = self.conv2(out)
        out += self.shortcut(x)
        out = self.activation(out)
        out = self.linear(out)
        return out
