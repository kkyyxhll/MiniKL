import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import timeit
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from models import MiniKLModel, MiniKLConfig, MiniKLBlock
import warnings
from collections import defaultdict
if __name__ == "__main__":

    def zipf_sentence_lengths(alpha: float, batch_size: int) -> torch.Tensor:
        sentence_lengths = np.empty(batch_size, dtype=int)
        for ibatch in range(batch_size):
            sentence_lengths[ibatch] = 1
            word = np.random.zipf(alpha)
            while word != 3 and word != 386 and word != 858:
                sentence_lengths[ibatch] += 1
                word = np.random.zipf(alpha)
        return torch.tensor(sentence_lengths)


    def gen_nested_data(config: MiniKLConfig, alpha: float = 1.2, batch_size: int = 64):
        lengths = zipf_sentence_lengths(alpha, batch_size)

        x = torch.nested.nested_tensor([
            torch.rand(seq_len, config.d_model) for seq_len in lengths
        ], layout=torch.jagged, )
        return x, lengths


    def benchmark(func, *args, **kwargs):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        begin = timeit.default_timer()
        output = func(*args, **kwargs)
        torch.cuda.synchronize()
        end = timeit.default_timer()
        return (end - begin), torch.cuda.max_memory_allocated() / 1e9


    class MiniKLModel(nn.Module):
        def __init__(self, config: MiniKLConfig):
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
            masked = None
            #masked = torch.triu(torch.ones(x.size(1), x.size(1)), diagonal=1).bool().to(x.device)
            for layer in self.layers:
                x = layer(x, masked=masked)
            output = self.fc(x)

            return output
    warnings.filterwarnings('ignore')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    nested_config = MiniKLConfig(is_nested=True)
    pad_config = MiniKLConfig()

    count = 40
    nested_times = []
    nested_memories = []
    pad_times = []
    pad_memories = []
    compile_times = []
    compile_memories = []
    len_count_dict = defaultdict()
    for i in range(count):
        nested_data, lengths  = gen_nested_data(nested_config)
        for length in lengths:
            length = int(length)
            if length in len_count_dict.keys():
                len_count_dict[length] += 1
            else :
                len_count_dict[length] = 1

        nested_data = nested_data.to(device)
        other_nested_data = nested_data.clone()
        nested_model = MiniKLModel(nested_config).to(device)
        pad_model = MiniKLModel(pad_config).to(device)

        torch.set_float32_matmul_precision('high')
        nested_output = benchmark(nested_model, nested_data)
        print(f"Nested_Tensor_Without_Compile: spend_time:{nested_output[0]:.5f}, gpu_memory:{nested_output[1]:.4f}GB")
        nested_times.append(nested_output[0])
        nested_memories.append(nested_output[1])
        compile_nested_model = torch.compile(nested_model)
        compile_nested_output = benchmark(compile_nested_model, other_nested_data)
        print(
            f"Nested_Tensor_With_Compile  : spend_time:{compile_nested_output[0]:.5f}, gpu_memory:{compile_nested_output[1]:.4f}GB")
        compile_times.append(compile_nested_output[0])
        compile_memories.append(compile_nested_output[1])
        pad__data = torch.nested.to_padded_tensor(nested_data, 5.0)
        pad_output = benchmark(pad_model, pad__data)
        print(f"Pad_Mask_Tensor spend_time  : spend_time:{pad_output[0]:.5f}, gpu_memory:{pad_output[1]:.4f}GB")
        pad_times.append(pad_output[0])
        pad_memories.append(pad_output[1])

    x = [i for i in range(count)]
    plt.figure()
    plt.plot(x, nested_times, color="r", label="nested_tensor")
    plt.plot(x, compile_times, color="yellow", label="nested_tensor_with_compile")
    plt.plot(x, pad_times, color="b", label="pad_tensor")
    plt.title("Nested | Pad Tensor Spent Time")
    plt.xlabel("count")
    plt.ylabel("time(s)")
    plt.legend()
    plt.savefig("../images/nested_pad_time.png")


    plt.figure()
    plt.plot(x, nested_memories, color="r", label="nested_tensor")
    plt.plot(x, compile_memories, color="yellow", label="nested_tensor_with_compile")
    plt.plot(x, pad_memories, color="b", label="pad_tensor")
    plt.title("Nested | Pad Tensor Spent GPU Memory")
    plt.xlabel("count")
    plt.ylabel("memory(GB)")
    plt.legend()
    plt.savefig("../images/nested_pad_memory.png")

    plt.figure()
    plt.bar(len_count_dict.keys(), len_count_dict.values())
    plt.title("Seq Length | Count")
    plt.ylabel("count")
    plt.xlabel("seq_len")
    plt.savefig("../images/seq_len.png")