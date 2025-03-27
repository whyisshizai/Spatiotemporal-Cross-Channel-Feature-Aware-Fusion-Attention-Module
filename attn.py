import math
import torch
import torch.nn as nn
from torch.nn.functional import scaled_dot_product_attention as sdp

class AttentionBlock(nn.Module):
    __doc__ = r"""
    IN:
        x: (B, in_channels, T, H, W)
        norm : group_norm
        num_groups (int): 默认: 32
    OUT:
        (B, in_channels, T,  H, W+)
    """
    def __init__(self, in_channels,num_groups=32,device=None):
        super().__init__()
        self.device = device
        self.in_channels = in_channels
        # 获取归一化层
        self.norm = nn.GroupNorm(num_groups,in_channels)
        # 空间QKV和投影
        self.to_qkv_spatial = nn.Conv3d(in_channels, in_channels * 3, 1)  #空间Q,K,V
        # 时间QKV和投影
        self.to_qkv_temporal = nn.Conv1d(in_channels, in_channels * 3, 1)  # 时间维度的QKV
    def forward(self, x, prompt_emb=None):
        b, c, t, h, w = x.shape
        x_spatial = self.norm(x)
        # 提取feature map
        avg_pool = nn.AdaptiveAvgPool3d((t, 1, 1)).to(self.device)
        f = avg_pool(x_spatial).reshape(b, c,-1).transpose(-1,-2)
        #b t c
        #感知特征提取器
        fc = nn.Sequential(
            nn.Linear(c, c//2), #压缩
            nn.ReLU(inplace=True),
            nn.Linear(c//2, c), #恢复
            nn.Sigmoid()  # 得到0-1的权重
        ).to(self.device)
        fm = fc(f).transpose(-1,-2).view(b,c,t,1,1)
        #空间过三维卷积
        qkv_s = self.to_qkv_spatial(x_spatial)  # 一次性生成QKV
        qkv_s = qkv_s.reshape(b, 3, self.in_channels, t, h, w)
        qs, ks, vs = qkv_s[:, 0], qkv_s[:, 1], qkv_s[:, 2]
        qs = qs.reshape(b, c, t, -1)  # [b, c, t,h*w]
        ks = ks.reshape(b, c, t, -1)
        vs = vs.reshape(b, c, t, -1)
        #b c h*w t
        # 时间过一维卷积得到单一维度
        qkv_t = self.to_qkv_temporal(x_spatial.reshape(b, c, -1))
        qkv_t = qkv_t.reshape(b, 3, self.in_channels, -1)
        #bs c t*h*w
        qt, kt, vt = qkv_t[:, 0], qkv_t[:, 1], qkv_t[:, 2]
        scale = 1.0 / math.sqrt(t)
        if prompt_emb is not None:
            #cross
            p_l= nn.Linear(prompt_emb.shape[-1],t * h * w).to(self.device)
            prompt = p_l(prompt_emb)  # [b, t*h*w]
            scale = 1.0 / math.sqrt(prompt.shape[-1])
            #q * k * v
            kt = sdp(kt, prompt, prompt, attn_mask=None, dropout_p=0.1)
        #cross
        #时间维度的k去和空间的q查询
        s1 = sdp(qs,kt.reshape(b, c, t, -1),vs).transpose(-2, -1).reshape(b, c, t, h, w)
        #时间的q去和空间上的k查询
        s2 = sdp(qt,ks.reshape(b, c, -1),vt).reshape(b, c, t, h, w)
        #residual block
        #回归到时间-时间，空间-空间，整体残差
        o = (s1 + s2) * fm +  x
        return o

s = torch.randn((12, 256, 4, 6, 6)).to("cuda")


#prompt_emb
#[b,seq_l]
p = torch.randn((1,512)).to("cuda")

attn = AttentionBlock(256,num_groups=32,device="cuda").to("cuda")
s = attn(s,p)
print(s.size())