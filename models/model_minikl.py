import torch
import torch.nn as nn
import torch.nn.functional as F

import math

class  MiniKLConfig:
    def __init__(self,
                 max_seq_len: int = 32 * 1024,  # RoPE预设的最大序列长度
                 d_model: int = 512,  # 模型隐层维度
                 base: float = float(1e6),  # RoPE base
                 num_heads: int = 8,  # attention层头数
                 g: int = 2,  # g=1 MQA 1<g<num_heads GQA g=num_heads MHA
                 ffn: str = "GLU",  # "GLU" | "FFN" FFN为标准Transformer的FFN， GLU为门控线性单元，默认GLU。
                 num_layers: int = 4,
                 flag_kv_cache: bool = False,  # 是否推理
                 vocab_size: int = 5000,
                 ):
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.base = base
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.g = g
        self.ffn = ffn
        self.num_layers = num_layers
        self.vocab_size = vocab_size

        assert num_heads % g == 0
        assert ffn == "GLU" or ffn == "FFN"


class RoPE(nn.Module):
    def __init__(self, config:  MiniKLConfig):
        super(RoPE, self).__init__()
        self.d_model = config.d_head
        self.base = config.base
        self.max_seq_len = config.max_seq_len
        self._pre_sin_cos_rope()

    def _pre_sin_cos_rope(self, ):
        d_model = self.d_model
        base = self.base
        max_seq_len = self.max_seq_len

        pos = torch.arange(max_seq_len).unsqueeze(-1)
        i = torch.arange(d_model // 2)
        omega = torch.exp(- 2 * i / d_model * math.log(base))

        rope_sin = torch.sin(pos * omega).repeat_interleave(2, dim=-1)
        rope_cos = torch.cos(pos * omega).repeat_interleave(2, dim=-1)

        self.register_buffer('rope_sin', rope_sin)
        self.register_buffer('rope_cos', rope_cos)

    def forward(self, x):
        """
        :param x: (batch_size, seq_len, d_model)
        :return:
        """
        batch_size, seq_len, d_model = x.size()
        assert d_model == self.d_model
        assert seq_len < self.max_seq_len

        cos_x = x
        sin_x = torch.stack([-x[..., 1::2], x[..., 0::2]], dim=-1).flatten(-2)
        return cos_x * self.rope_cos[:seq_len, :] + sin_x * self.rope_sin[:seq_len, :]


class Attention(nn.Module):
    def __init__(self, config:  MiniKLConfig):
        super(Attention, self).__init__()
        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.d_head = config.d_head
        self.g = config.g

        self.rope = RoPE(config=config)

        self.w_q = nn.Linear(self.d_model, self.d_model)
        self.w_k = nn.Linear(self.d_model, self.d_head * self.g)
        self.w_v = nn.Linear(self.d_model, self.d_head * self.g)
        self.w_o = nn.Linear(self.d_model, self.d_model)

    def forward(self, x, masked=None):
        """
        :param x: (batch_size, seq_len, d_model)
        :param masked: torch.triu(torch.ones_like(torch.rand(batch_size*num_heads, seq_len, seq_len)), diagonal=1).bool()
        :return:
        """
        q = self._mapper(self.w_q(x), g=self.num_heads)
        k = self._mapper(self.w_k(x), g=self.g)
        v = self._mapper(self.w_v(x), g=self.g)

        q = self.rope(q)
        k = self.rope(k)

        attention_score = q @ k.transpose(-2, -1)
        if masked is not None:
            attention_score[masked] = -float('inf')
        attention = F.softmax(attention_score / math.sqrt(self.d_head), dim=-1) @ v
        attention = self._reduce(attention)
        attention = self.w_o(attention)
        return attention

    def _mapper(self, x, g):
        """
        :param x: (batch_size, seq_len, d_head * g)
        :return: (batch_size*num_heads, seq_len, d_head)
        """
        batch_size, seq_len, _ = x.size()
        x = x.reshape(batch_size, seq_len, g, self.d_head)
        x = x.permute(0, 2, 1, 3).reshape(batch_size * g, seq_len, self.d_head)
        x = x.repeat_interleave(self.num_heads // g, dim=0)
        return x

    def _reduce(self, x):
        """
        :param x: (batch_size*num_heads, seq_len, d_model//num_heads)
        :return: (batch_size, seq_len, d_model)
        """
        batch_size, seq_len = x.size(0) // self.num_heads, x.size(1)
        x = x.reshape(batch_size, self.num_heads, seq_len, self.d_head)
        x = x.permute(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)
        return x


class FFN(nn.Module):
    def __init__(self, config:  MiniKLConfig):
        super(FFN, self).__init__()
        self.d_model = config.d_model
        self.d_ff = config.d_model * 4

        self.w_up = nn.Linear(self.d_model, self.d_ff)
        self.w_down = nn.Linear(self.d_ff, self.d_model)

    def forward(self, x):
        """
        :param x: (batch_size, seq_len, d_model)
        :return:
        """
        up_x = self.w_up(x)
        down_x = self.w_down(up_x)
        return down_x


class GLU(nn.Module):
    def __init__(self, config:  MiniKLConfig):
        super(GLU, self).__init__()
        self.d_model = config.d_model
        self.d_ff = 2 * 4 * self.d_model // 3
        self.w_up_a = nn.Linear(self.d_model, self.d_ff, )
        self.w_up_b = nn.Linear(self.d_model, self.d_ff)
        self.w_down = nn.Linear(self.d_ff, self.d_model)

    def forward(self, x):
        a = self.w_up_a(x)
        b = self.w_up_b(x)
        up_x = a * self._swish(b)
        down_x = self.w_down(up_x)
        return down_x

    @staticmethod
    def _swish(x):
        """
        :param x: (batch_size, seq_len, d_ff)
        :return:
        """
        return F.sigmoid(x) * x


class MiniKLBlock(nn.Module):
    def __init__(self, config:  MiniKLConfig):
        super(MiniKLBlock, self).__init__()
        self.attention = Attention(config=config)
        self.d_model = config.d_model
        if config.ffn == "GLU":
            self.ffn = GLU(config=config)
        elif config.ffn == "FFN":
            self.ffn = FFN(config=config)
        self.norm1 = nn.LayerNorm(self.d_model)
        self.norm2 = nn.LayerNorm(self.d_model)

    def forward(self, x, masked=None):
        x = x + self.attention(self.norm1(x), masked=masked)
        x = x + self.ffn(self.norm2(x))
        return x


class MiniKLModel(nn.Module):
    def __init__(self, config:  MiniKLConfig):
        super(MiniKLModel, self).__init__()
        self.num_heads = config.num_heads
        self.d_model = config.d_model
        self.vocab_size = config.vocab_size
        self.embeddings = nn.Embedding(self.vocab_size, self.d_model)
        self.layers = nn.ModuleList([MiniKLBlock(config=config) for _ in range(config.num_layers)])
        self.fc = nn.Linear(self.d_model, self.vocab_size)

    def forward(self, x, ):
        """
        :param x: (batch_size, seq_len)
        :return:
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        batch_size, seq_len,  = x.size()
        x = self.embeddings(x)
        masked = torch.triu(torch.ones(batch_size * self.num_heads, seq_len, seq_len), diagonal=1).bool()
        for layer in self.layers:
            x = layer(x, masked=masked)
        output = self.fc(x)

        return output