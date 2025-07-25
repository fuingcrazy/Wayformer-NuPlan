import math
from typing import List, Dict,Optional

import torch
import torch.nn as nn

class FourierEmbedding(nn.Module):

    def __init__(self,input_dim:int, hidden_dim:int, num_freq_bands:int) -> None:
        super(FourierEmbedding,self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.freqs = nn.Embedding(input_dim,num_freq_bands)if input_dim != 0 else None    #可学习参数矩阵
        self.mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(num_freq_bands * 2 + 1, hidden_dim),   #cos,sin加上原始值
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden_dim,hidden_dim),
                )
                for _ in range(input_dim)
            ]
        )
        self.to_out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim,hidden_dim),
        )
    
    def forward(self, continuous_inputs: Optional[torch.Tensor]):
        x = continuous_inputs.unsqueeze(-1) * self.freqs.weight * 2 * math.pi    #按照频率缩放
        x = torch.cat([x.cos(),x.sin(),continuous_inputs.unsqueeze(-1)],dim=-1)   #拼接cos,sin和原始值,维度[B, T, D, 2 * num_freq_bands + 1]
        continuous_embs : List[Optional[torch.Tensor]] = [None] * self.input_dim
        for i in range(self.input_dim):
            continuous_embs[i] = self.mlps[i](x[...,i,:])   #每个维度的MLP处理
        x = torch.stack(continuous_embs).sum(dim=0)   #将所有维度的结果相加
        x = self.to_out(x)   #输出层
        return x   #返回[B, T, D, hidden_dim]的张量

