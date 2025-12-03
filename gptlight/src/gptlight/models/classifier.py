from __future__ import annotations

from typing import Any, Mapping

import torch.nn as nn

from gptlight.config import GPTConfig
from .base import GPTModel

class GPTClassifier(GPTModel):
    """
    A GPT-based model adapted for supervised classification tasks.
    """

    def __init__(self, cfg: GPTConfig, num_classes: int = 2):
        super().__init__(cfg)

        self.cfg = cfg
        self.num_classes = num_classes

        for param in self.parameters():
            param.requires_grad = False
        
        self.out_head = nn.Linear(self.cfg.emb_dim, num_classes)

        for param in self.trf_blocks[-1].parameters():
            param.requires_grad = True

        for param in self.final_norm.parameters():
            param.requires_grad = True
