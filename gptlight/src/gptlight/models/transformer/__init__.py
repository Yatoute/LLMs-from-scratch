from __future__ import annotations

import torch
import torch.nn as nn

from gptlight.models.transformer.normalization import LayerNorm
from gptlight.models.transformer.attention import MultiHeadAttention
from gptlight.models.transformer.ffn import FeedForward

from gptlight.config import GPTConfig

class GPTTransformerBlock(nn.Module):
    
    def __init__(self, cfg:GPTConfig):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg.emb_dim,
            d_out=cfg.emb_dim,
            num_heads=cfg.n_heads,
            context_length=cfg.context_length,
            dropout=cfg.drop_rate,
            qkv_bias=cfg.qkv_bias
        )
        
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(emb_dim=cfg.emb_dim)
        self.norm2 = LayerNorm(emb_dim=cfg.emb_dim)
        self.drop_shortcut = nn.Dropout(cfg.drop_rate)
    
    def forward(self, x:torch.Tensor):
        
        shortcut =  x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x) 
        x = x + shortcut
        
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        
        return x     
        
