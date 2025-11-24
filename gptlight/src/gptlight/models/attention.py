from __future__ import annotations

import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    
    def __init__(self, d_in:int, d_out:int, qkv_bias:bool=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        
    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_key(x)
        values = self.W_value(x)
        attn_scores = queries@keys.T
        attn_weights = torch.softmax(attn_scores/keys.shape[-1]**0.5, dim=-1)
        context_vecs = attn_weights@values
        return context_vecs
        