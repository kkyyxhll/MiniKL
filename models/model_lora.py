import torch
import torch.nn as nn
import torch.nn.functional as F

import math

class LoRA(nn.Module):
    def __init__(self, rank, in_features, out_features):
        super().__init__()
        self.A = nn.Linear(in_features, rank, bias=False)
        self.B = nn.Linear(rank, out_features, bias=False)
        self.A.weight.data.normal_(0, 0.02)
        self.B.weight.data.zero_()

    def forward(self, x):
        """
        :param x: (batch_size, seq_len, d_model)
        :return:
        """
        return self.B(self.A(x))

def apply_lora(model:nn.Module, rank=8):

    for n, m in model.named_modules():
        if isinstance(m, nn.Linear):
            in_features, out_features = m.weight.data.shape
            lora = LoRA(rank, in_features, out_features)
            setattr(m, "lora", lora)
            base_forward = m.forward
            def forward_with_lora(x, layer_1:base_forward, layer_2:lora):
                return layer_1(x) + layer_2(x)

            m.forward = forward_with_lora


if __name__ == "__main__":
    from models import MiniKLModel, MiniKLConfig
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = MiniKLModel(MiniKLConfig())
    apply_lora(model)
    print(model)