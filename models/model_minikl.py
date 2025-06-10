import torch
import torch.nn as nn
import torch.nn.functional as F

import math



class MiniKLConfig:
    def __init__(self,
                 max_seq_len: int = 32 * 1024,  # RoPE预设的最大序列长度
                 d_model: int = 512,  # 模型隐层维度
                 theta: float = float(1e6),  # RoPE base
                 num_heads: int = 8,  # attention层头数
                 g: int = 8,  # g=1 MQA 1<g<num_heads GQA g=num_heads MHA
                 ffn: str = "GLU",  # "GLU" | "FFN" FFN为标准Transformer的FFN， GLU为门控线性单元，默认GLU。
                 num_layers: int = 4,
                 flag_kv_cache: bool = False,  # 是否推理
                 vocab_size: int = 5000,
                 is_nested: bool = False, # 是否使用嵌套张量
                 ):
        assert is_nested==False or (is_nested==True and num_heads == g) # is_nested 目前只支持 MHA,暂不支持 GQA
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.theta = theta
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.g = g
        self.ffn = ffn
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.is_nested = is_nested
        assert num_heads % g == 0
        assert ffn == "GLU" or ffn == "FFN"


class RoPE(nn.Module):
    def __init__(self, config:MiniKLConfig):
        super().__init__()
        self.theta = config.theta
        self.max_seq_len = config.max_seq_len
        self.d_model = config.d_head
        self._pre_cal()
    def _pre_cal(self):
        pos = torch.arange(self.max_seq_len).unsqueeze(-1) #(max_seq_len, 1)
        i = torch.arange(self.d_model//2)
        omega = torch.exp(- 2 * i / self.d_model * math.log(self.theta)) #(d_model//2)
        rope_sin = torch.sin(pos * omega).repeat_interleave(2, -1)
        rope_cos = torch.cos(pos * omega).repeat_interleave(2, -1)
        self.register_buffer("rope_sin", rope_sin)
        self.register_buffer("rope_cos", rope_cos)
    def forward(self, x):
        """
            x:(batch_size, num_head, seq_len, d_head) | (batch_size, num_head, seq_len*, d_head)
        """
        if type(x) == torch.Tensor:# norm tensor
            return (x * self.rope_cos[:x.size(-2), :] +
                    torch.stack([x[..., 1::2], -x[..., 0::2]], dim=-1).flatten(-2, -1) * self.rope_sin[:x.size(-2), :])
        else : # torch.nested.nested_tensor
            values = torch.rand(x.size(1), 0, x.size(-1)).to(x.device)
            for nt in x:
                rope_nt = nt * self.rope_cos[:nt.size(-2), :] + torch.stack([nt[..., 1::2], -nt[..., 0::2]], dim=-1).flatten(-2, -1)*self.rope_sin[:nt.size(-2),:]
                values = torch.cat([values, rope_nt], dim=-2)
            output = torch.nested.nested_tensor_from_jagged(values=values, offsets=x.offsets(), jagged_dim=2)

            return output


class Attention(nn.Module):
    def __init__(self, config:  MiniKLConfig):
        super(Attention, self).__init__()
        self.is_nested = config.is_nested
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
        # q = self._mapper(self.w_q(x), g=self.num_heads)
        # k = self._mapper(self.w_k(x), g=self.g)
        # v = self._mapper(self.w_v(x), g=self.g)
        #
        # q = self.rope(q)
        # k = self.rope(k)
        #
        # attention_score = q @ k.transpose(-2, -1)
        # if masked is not None:
        #     attention_score.masked_fill_(masked, float("-inf"))
        # attention = F.softmax(attention_score / math.sqrt(self.d_head), dim=-1) @ v
        # attention = self._reduce(attention)
        q = self.w_q(x).unflatten(-1, [self.num_heads, self.d_head]).transpose(1, 2)
        q = self.rope(q)
        k = self.w_k(x).unflatten(-1, [self.g, self.d_head]).transpose(1, 2)
        k = self.rope(q)
        v = self.w_v(x).unflatten(-1, [self.g, self.d_head]).transpose(1, 2)
        if not self.is_nested:
            k = k.repeat_interleave(self.num_heads//self.g, 1)
            v = v.repeat_interleave(self.num_heads//self.g, 1)
        attention = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).flatten(-2, -1)

        attention = self.w_o(attention)
        return attention

    def _mapper(self, x, g):
        """
        :param x: (batch_size, seq_len, d_head * g)
        :return: (batch_size*num_heads, seq_len, d_head)
        """
        batch_size, seq_len, _ = x.size()
        x = x.reshape(batch_size, seq_len, g, self.d_head)
        x = x.permute(0, 2, 1, 3).reshape(batch_size*g, seq_len, self.d_head)
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
        self.norm1 = nn.RMSNorm(self.d_model)
        self.norm2 = nn.RMSNorm(self.d_model)

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

        x = self.embeddings(x)
        masked = torch.triu(torch.ones(x.size(1), x.size(1)), diagonal=1).bool().to(x.device)
        for layer in self.layers:
            x = layer(x, masked=masked)
        output = self.fc(x)

        return output


