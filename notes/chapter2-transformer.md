# 第2章：Transformer 架构

## 2.1 注意力机制

核心代码实现见：[attention_demo.py](../code/chapter2-transformer/attention_demo.py)

关键公式：

$$Attention(Q,K,V) = softmax(\frac{QK^T}{\sqrt{d_k}})V$$