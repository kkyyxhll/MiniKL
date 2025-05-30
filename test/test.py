import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np

import timeit

class Config:
    def __init__(self,
                 d_model: int=512,
                 num_head: int=8,
                 theta: float=1e6,
                 max_seq_len: int=512):
        self.d_model = d_model
        self.num_head = num_head
        self.d_head = self.d_model // self.num_head
        self.theta = theta
        self.max_seq_len = max_seq_len

class Attention(nn.Module):
    def __init__(self, config:Config):
        super().__init__()
        self.d_model = config.d_model
        self.num_head = config.num_head
        self.d_head = config.d_head

        self.w_q = nn.Linear(self.d_model, self.d_model)
        self.w_k = nn.Linear(self.d_model, self.d_model)
        self.w_v = nn.Linear(self.d_model, self.d_model)
        self.w_o = nn.Linear(self.d_model, self.d_model)

    def forward(self, x, masked:torch.bool=None): # (batch_size, seq_len, d_model)

        q = self.w_q(x).unflatten(-1, [self.num_head, self.d_head]).transpose(1, 2)
        k = self.w_k(x).unflatten(-1, [self.num_head, self.d_head]).transpose(1, 2)
        v = self.w_v(x).unflatten(-1, [self.num_head, self.d_head]).transpose(1, 2)

        attention_score = q @ (k.transpose(-1, -2))

        if masked is None:
            masked = torch.triu(torch.ones_like(attention_score[0, 0]), diagonal=1).bool()
        attention_score.masked_fill_(masked, float("-inf"))

        attention_score = F.softmax(attention_score / math.sqrt(self.d_head), -1)

        attention = (attention_score @ v ).transpose(1, 2).flatten(-2, -1)
        attention = self.w_o(attention)
        return attention

def zipf_texts_lengths(batch_size:int=16, alpha:float=1.2, max_seq_len:int=512) -> torch.Tensor:
    text_lengths = np.zeros(batch_size, dtype=int)
    for i in range(batch_size):
        text_lengths[i] = 1
        word = np.random.zipf(alpha)
        while word != 3 and word != 386 and word != 858 and text_lengths[i] < max_seq_len:
            text_lengths[i] += 1
            word = np.random.zipf(alpha)
    return torch.tensor(text_lengths)

def generate_batch_data(batch_size, config: Config):
    texts_lengths = zipf_texts_lengths(batch_size=batch_size)

    d_model = config.d_model
    x = torch.nested.nested_tensor(
        [torch.rand(length.item(), d_model) for length in texts_lengths],
        layout=torch.jagged
    )

    return x


def benchmark(func, *args, **kwargs):
    torch.cuda.synchronize()
    begin = timeit.default_timer()
    output = func(*args, **kwargs)
    torch.cuda.synchronize()
    end = timeit.default_timer()
    return output, (end-begin), torch.cuda.max_memory_allocated()

if __name__ == "__main__":
    config = Config()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 42
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)

    attention_layer = Attention(config).to(device)

    batch_size = 128

    nested_data = generate_batch_data(batch_size, config).to(device)

    pad_data = torch.nested.to_padded_tensor(nested_data, 0.0).to(device)

    output, time, memory = benchmark(attention_layer, pad_data)

    output, time, memory = benchmark(attention_layer, nested_data)

