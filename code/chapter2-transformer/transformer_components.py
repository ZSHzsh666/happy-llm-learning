import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ------------------- 1. 注意力机制核心函数 -------------------
def attention(query, key, value, mask=None, dropout=None):
    """
    计算缩放点积注意力。
    Args:
        query: (batch_size, n_heads, seq_len, d_k)
        key:   (batch_size, n_heads, seq_len, d_k)
        value: (batch_size, n_heads, seq_len, d_v)
        mask:  (batch_size, 1, seq_len, seq_len) or None
        dropout: nn.Dropout 实例
    """
    d_k = query.size(-1)
    # 1. 计算 Q 和 K 的点积，并缩放
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    # 2. 应用掩码（如果有）
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    # 3. Softmax 归一化得到注意力权重
    p_attn = F.softmax(scores, dim=-1)

    # 4. 应用 Dropout
    if dropout is not None:
        p_attn = dropout(p_attn)

    # 5. 对 value 加权求和
    return torch.matmul(p_attn, value), p_attn


# ------------------- 2. 多头注意力模块 -------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, n_heads=8, dropout=0.1, is_causal=False, max_seq_len=1024):
        super().__init__()
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.is_causal = is_causal

        # 定义 Q, K, V 的线性投影层
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        
        # 定义输出的线性投影层
        self.wo = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        # 如果是因果的（用于Decoder），注册一个上三角掩码
        if self.is_causal:
            mask = torch.full((1, 1, max_seq_len, max_seq_len), float("-inf"))
            mask = torch.triu(mask, diagonal=1)
            self.register_buffer("mask", mask)

    def forward(self, query, key, value):
        batch_size = query.size(0)
        
        # 1. 线性投影并拆分成多头
        # 形状变换: (B, T, d_model) -> (B, n_heads, T, d_k)
        Q = self.wq(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.wk(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.wv(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # 2. 计算注意力
        if self.is_causal:
            seq_len = query.size(1)
            # 使用注册好的因果掩码
            mask = self.mask[:, :, :seq_len, :seq_len] == float("-inf") 
            x, _ = attention(Q, K, V, mask=mask, dropout=self.dropout)
        else:
            x, _ = attention(Q, K, V, dropout=self.dropout)

        # 3. 拼接多头输出并做最终投影
        # 形状变换: (B, n_heads, T, d_k) -> (B, T, d_model)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.wo(x)


# ------------------- 3. 前馈神经网络模块 -------------------
class FeedForward(nn.Module):
    def __init__(self, d_model=512, d_ff=2048, dropout=0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w2(self.dropout(F.relu(self.w1(x))))


# ------------------- 4. 层归一化模块 -------------------
class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


# ------------------- 5. 位置编码模块 -------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        # 计算分母中的 10000^(2i/d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        # 计算正弦和余弦
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 增加 batch 维度，并注册为 buffer（不参与训练）
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        return x + self.pe[:, :x.size(1)].requires_grad_(False)


# ------------------- 6. Encoder Layer -------------------
class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.norm1 = LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout, is_causal=False)
        self.norm2 = LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)

    def forward(self, x):
        # Pre-Norm 结构
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x

# 代码接上一节，需导入之前定义的 MultiHeadAttention, FeedForward, LayerNorm

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.norm1 = LayerNorm(d_model)
        # Encoder 的自注意力不需要掩码，is_causal=False
        self.attn = MultiHeadAttention(d_model, n_heads, dropout, is_causal=False)
        self.norm2 = LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)

    def forward(self, x):
        # 子层1: Self-Attention
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        # 子层2: Feed Forward
        x = x + self.ff(self.norm2(x))
        return x

class Encoder(nn.Module):
    def __init__(self, n_layers, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout) 
            for _ in range(n_layers)
        ])
        self.norm = LayerNorm(d_model) # 最终层归一化

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)
    
class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.norm1 = LayerNorm(d_model)
        # 第一个注意力层：带掩码的自注意力
        self.masked_attn = MultiHeadAttention(d_model, n_heads, dropout, is_causal=True)
        
        self.norm2 = LayerNorm(d_model)
        # 第二个注意力层：交叉注意力 (Q来自Decoder, K,V来自Encoder)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout, is_causal=False)
        
        self.norm3 = LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)

    def forward(self, x, enc_out):
        # 子层1: 掩码自注意力
        x = x + self.masked_attn(self.norm1(x), self.norm1(x), self.norm1(x))
        
        # 子层2: 交叉注意力 (注意参数传递)
        x = x + self.cross_attn(self.norm2(x), enc_out, enc_out)
        
        # 子层3: 前馈网络
        x = x + self.ff(self.norm3(x))
        return x

class Decoder(nn.Module):
    def __init__(self, n_layers, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout) 
            for _ in range(n_layers)
        ])
        self.norm = LayerNorm(d_model)

    def forward(self, x, enc_out):
        for layer in self.layers:
            x = layer(x, enc_out)
        return self.norm(x)

# ------------------- 简单测试 -------------------
if __name__ == "__main__":
    print("===== 测试 Transformer 核心组件 =====")
    
    # 超参数
    batch_size = 2
    seq_len = 10
    d_model = 512
    n_heads = 8
    
    # 创建随机输入
    x = torch.randn(batch_size, seq_len, d_model)
    
    # 测试多头注意力 (Encoder 模式)
    mha_enc = MultiHeadAttention(d_model, n_heads, is_causal=False)
    out_enc = mha_enc(x, x, x)
    print(f"Encoder MHA 输出形状: {out_enc.shape}") # 预期: [2, 10, 512]
    
    # 测试多头注意力 (Decoder 模式，带掩码)
    mha_dec = MultiHeadAttention(d_model, n_heads, is_causal=True)
    out_dec = mha_dec(x, x, x)
    print(f"Decoder MHA 输出形状: {out_dec.shape}") # 预期: [2, 10, 512]
    
    # 测试位置编码
    pe = PositionalEncoding(d_model)
    out_pe = pe(x)
    print(f"位置编码后形状: {out_pe.shape}") # 预期: [2, 10, 512]

    # 测试完整的 Encoder Layer
    enc_layer = EncoderLayer(d_model, n_heads, d_ff=2048)
    out_enc_layer = enc_layer(x)
    print(f"EncoderLayer 输出形状: {out_enc_layer.shape}") # 预期: [2, 10, 512]
    
    print("\n所有测试通过！")

    print("\n===== 测试 Encoder 和 Decoder =====")
    # 超参数
    batch_size = 2
    seq_len_src = 10 # 源序列长度
    seq_len_tgt = 8  # 目标序列长度
    d_model = 512
    n_heads = 8
    n_layers = 3
    d_ff = 2048

    # 模拟输入
    src = torch.randn(batch_size, seq_len_src, d_model)
    tgt = torch.randn(batch_size, seq_len_tgt, d_model)

    # 实例化 Encoder 和 Decoder
    encoder = Encoder(n_layers, d_model, n_heads, d_ff)
    decoder = Decoder(n_layers, d_model, n_heads, d_ff)

    # 前向传播
    enc_output = encoder(src)
    dec_output = decoder(tgt, enc_output)

    print(f"Encoder 输入形状: {src.shape}")
    print(f"Encoder 输出形状: {enc_output.shape}") # 预期: [2, 10, 512]
    print(f"Decoder 输入形状: {tgt.shape}")
    print(f"Decoder 输出形状: {dec_output.shape}") # 预期: [2, 8, 512]
    print("\nEncoder-Decoder 结构测试通过！")