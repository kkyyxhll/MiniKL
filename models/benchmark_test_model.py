import torch

from models import MiniKLModel, MiniKLConfig

import timeit


def benchmark(func, *args, **kwargs):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = timeit.default_timer()
    func(*args, *kwargs)
    torch.cuda.synchronize()
    end = timeit.default_timer()
    return end-start, torch.cuda.max_memory_allocated()

if __name__ == "__main__":
    batch_size = 16
    seq_len = 512
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)
    if device == "cuda":
        torch.cuda.manual_seed_all(42)

    model = MiniKLModel(MiniKLConfig()).to(device)

    x = torch.randint_like(torch.rand(batch_size, seq_len), high=100).long().to(device)

    test_time = 10
    total_time = 0.0
    total_memory = 0.0
    for _ in range(test_time):
        time, memory = benchmark(model, x)
        total_time += time
        total_memory += memory
    avg_time = total_time / test_time
    avg_memory = total_memory / test_time
    print(avg_time, avg_memory / (1024**3))

# 0.5680936810002095 2.133625030517578
# 0.09993730399983178 2.133924102783203