"""Small decoder-only (GPT-style) transformer over semantic-ID tokens."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils import PAD


class Block(nn.Module):
    def __init__(self, d, heads, d_ff, p):
        super().__init__()
        self.h, self.p = heads, p
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, d_ff), nn.GELU(), nn.Linear(d_ff, d))
        self.drop = nn.Dropout(p)

    def forward(self, x, mask):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, T, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                           dropout_p=self.p if self.training else 0.0)
        x = x + self.drop(self.proj(a.transpose(1, 2).reshape(B, T, D)))
        return x + self.drop(self.mlp(self.ln2(x)))


class SemanticIDTransformer(nn.Module):
    def __init__(self, vocab_size, tokens_per_item, max_len, d_model=256, n_layers=4,
                 n_heads=4, d_ff=1024, dropout=0.1):
        super().__init__()
        self.ti, self.vocab_size = tokens_per_item, vocab_size
        self.tok = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        self.pos = nn.Embedding(max_len, d_model)                  # absolute position in the sequence
        self.lvl = nn.Embedding(tokens_per_item + 1, d_model)      # which token of the item (+1 slot for BOS)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok.weight                         # weight tying
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, tokens, last_only=False):
        """tokens: [B,T], PAD=0 (left- OR right-padded). Returns logits [B,T,V] (or [B,V])."""
        B, T = tokens.shape
        valid = tokens != PAD
        pos = (valid.cumsum(1) - 1).clamp(min=0)                   # positions ignore padding
        lvl = torch.where(pos == 0, torch.full_like(pos, self.ti), (pos - 1) % self.ti)
        x = self.drop(self.tok(tokens) + self.pos(pos) + self.lvl(lvl))

        causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=tokens.device))
        eye = torch.eye(T, dtype=torch.bool, device=tokens.device)
        mask = (causal[None, None] & valid[:, None, None, :]) | eye[None, None]

        for blk in self.blocks:
            x = blk(x, mask)
        x = self.ln(x)
        if last_only:
            x = x[:, -1]
        return self.head(x)