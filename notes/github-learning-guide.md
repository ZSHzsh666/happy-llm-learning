# GitHub 学习记录指南

> 本文档记录如何使用 GitHub 来管理《Happy-LLM》的学习过程，包含 Markdown 语法速查和专业技巧。

---

## 一、Markdown 语法速查

### 1. 代码块（展示代码必备）

**语法**：用三个反引号包裹，并指定语言

```python
# 示例：Transformer 中的注意力机制
import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        self.n_head = n_head
        self.d_model = d_model
        assert d_model % n_head == 0, "d_model must be divisible by n_head"
        
        self.d_k = d_model // n_head
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
    
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        # ... 实现细节
        return output