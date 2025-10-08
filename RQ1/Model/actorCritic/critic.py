import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 方案1: 残差连接 + 注意力机制的评价网络
class CriticNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim=128, num_heads=4):
        super().__init__()
        # 特征提取
        self.embed = nn.Linear(state_dim, hidden_dim)
        # 自注意力层
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=0.1)
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        # 输出层
        self.fc_out = nn.Linear(hidden_dim, 1)
        
        # 初始化
        nn.init.xavier_uniform_(self.embed.weight)

    def forward(self, x):
        # x: [batch_size, state_dim]
        x = self.embed(x).unsqueeze(0)  # [1, batch_size, hidden_dim]
        attn_out, _ = self.attention(x, x, x)
        x = x + attn_out  # 残差
        x = self.ffn(x) + x
        return self.fc_out(x.squeeze(0))