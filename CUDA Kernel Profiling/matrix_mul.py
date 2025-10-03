# 1. Basic CUDA Kernel Profiling (Matrix Multiplication)

import torch
import torch.profiler as profiler

# Simple operation to profile
def simple_op():
    x = torch.randn(2048, 2048, device='mps')
    y = torch.randn(2048, 2048, device='mps')
    z = x @ y
    return y

with profiler.profile(
    activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True
) as prof:
    simple_op()


# Print summary
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
