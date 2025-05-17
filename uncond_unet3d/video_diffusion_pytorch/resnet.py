import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange

def exists(x):
    return x is not None

# building block modules note 3d convlution here
# using GroupNorm instead of RMSNorm    
class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.proj = nn.Conv3d(dim, dim_out, (1, 3, 3), padding = (0, 1, 1))
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
 
        return self.act(x)

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, emb_dim = None, groups = 8, frames=16):
        """
        :param dim: input channel
        :param dim_out:  output channel
        :param emb_dim: extra embedding to fuse, such as time and index or control
        :param groups: group for conv3d
        """ 
        super().__init__()
        self.frames = frames
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, dim_out * 2)
        ) if exists(emb_dim) else None

        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)
        self.res_conv = nn.Conv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, emb = None):
        scale_shift = None
        b = x.shape[0]//self.frames if exists(emb) else None
        if exists(self.mlp):
            assert exists(emb), 'time emb must be passed in'
            emb = self.mlp(emb)
            # TODO: revise the emb to [b c f 1 1] format for broadcast; Note the conv3d here
            emb = rearrange(emb, '(b f) c -> b c f 1 1', b=b, f=self.frames)
    
            scale_shift = emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)