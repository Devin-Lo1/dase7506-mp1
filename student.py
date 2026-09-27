"""RoPE + SwiGLU causal Transformer; learned positions provide an ablation.
References: arXiv:2104.09864 and arXiv:2002.05202.
No temporary state survives a forward call.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config['width']
        self.heads = config['heads']
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.proj = nn.Linear(width, width, bias=False)
        hidden = math.ceil((8 * width / 3) / 32) * 32
        self.gate_up = nn.Linear(width, 2 * hidden, bias=False)
        self.down = nn.Linear(hidden, width, bias=False)
        self.dropout = nn.Dropout(config.get('dropout', 0.1))
        self.attention_dropout = config.get('attention_dropout', 0.0)

    @staticmethod
    def rotate(x, cos, sin):
        even, odd = x[..., ::2], x[..., 1::2]
        return torch.stack((even * cos - odd * sin,
                            even * sin + odd * cos), dim=-1).flatten(-2)

    def forward(self, x, rotary=None):
        batch, length, width = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(
            batch, length, 3, self.heads, width // self.heads
        ).permute(2, 0, 3, 1, 4)
        if rotary is not None:
            q = self.rotate(q, *rotary)
            k = self.rotate(k, *rotary)
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True,
            dropout_p=self.attention_dropout if self.training else 0.0)
        x = x + self.dropout(self.proj(attended.transpose(1, 2).reshape(batch, length, width)))
        gate, up = self.gate_up(self.norm2(x)).chunk(2, dim=-1)
        return x + self.dropout(self.down(F.silu(gate) * up))


class StudentGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        self.context = config['context']
        width, heads = config['width'], config['heads']
        if width % heads or (width // heads) % 2:
            raise ValueError('Each attention head must have an even width.')
        self.token = nn.Embedding(config['vocab'], width)
        self.embedding_dropout = nn.Dropout(config.get('embedding_dropout', 0.0))
        position = config.get('positional_encoding', 'rotary')
        if position not in ('rotary', 'learned'):
            raise ValueError('Unknown positional encoding.')
        self.pos = nn.Embedding(self.context, width) if position == 'learned' else None
        head_width = width // heads
        angles = torch.outer(torch.arange(self.context).float(),
                             10000.0 ** (-torch.arange(0, head_width, 2).float() / head_width))
        self.register_buffer('rotary_cos', angles.cos(), persistent=False)
        self.register_buffer('rotary_sin', angles.sin(), persistent=False)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config['depth'])])
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, config['vocab'], bias=False)
        self.apply(self.initialize)
        self.head.weight = self.token.weight
        for block in self.blocks:
            nn.init.normal_(block.proj.weight, std=0.02 / math.sqrt(2 * config['depth']))
            nn.init.normal_(block.down.weight, std=0.02 / math.sqrt(2 * config['depth']))

    @staticmethod
    def initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if getattr(module, 'bias', None) is not None:
                nn.init.zeros_(module.bias)

    def features(self, ids):
        length = ids.shape[1]
        if length > self.context:
            raise ValueError('Input exceeds the fixed context.')
        x = self.embedding_dropout(self.token(ids))
        rotary = None
        if self.pos is not None:
            x = x + self.pos(torch.arange(length, device=ids.device))
        else:
            rotary = (self.rotary_cos[:length], self.rotary_sin[:length])
        for block in self.blocks:
            x = block(x, rotary)
        return self.norm(x)

    def forward(self, ids):
        return self.head(self.features(ids))

    def predict_log_probs(self, ids):
        return F.log_softmax(self(ids).float(), dim=-1)


def build_model(config):
    return StudentGPT(config)
