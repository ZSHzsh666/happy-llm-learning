"""
Transformer 注意力机制演示代码
学习目标：理解 Scaled Dot-Product Attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """缩放点积注意力"""
    
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: [batch, n_head, seq_len, d_k]
            key:   [batch, n_head, seq_len, d_k]
            value: [batch, n_head, seq_len, d_v]
            mask:  [batch, 1, seq_len, seq_len] or None
        """
        d_k = query.size(-1)
        
        # 1. 计算注意力分数: Q @ K^T / sqrt(d_k)
        scores = torch.matmul(query, key.transpose(-2, -1)) / (d_k ** 0.5)
        
        # 2. 应用 mask（如果有）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 3. Softmax 得到注意力权重
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 4. 加权求和
        output = torch.matmul(attn_weights, value)
        
        return output, attn_weights


# 测试代码
if __name__ == "__main__":
    batch_size = 2
    n_head = 8
    seq_len = 10
    d_k = 64
    d_v = 64
    
    # 创建随机输入
    Q = torch.randn(batch_size, n_head, seq_len, d_k)
    K = torch.randn(batch_size, n_head, seq_len, d_k)
    V = torch.randn(batch_size, n_head, seq_len, d_v)
    
    # 创建 attention 层
    attention = ScaledDotProductAttention()
    
    # 前向传播
    output, weights = attention(Q, K, V)
    
    print(f"输入 Q shape: {Q.shape}")
    print(f"输出 shape: {output.shape}")
    print(f"注意力权重 shape: {weights.shape}")
    print(f"\n第一个 batch 第一个 head 的注意力权重:\n{weights[0, 0]}")