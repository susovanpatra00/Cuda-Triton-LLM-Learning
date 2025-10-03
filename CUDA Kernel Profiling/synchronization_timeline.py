import torch
import torch.profiler as profiler

# A function with CPU–GPU sync
def sync_op():
    x = torch.randn(4096, 4096, device="cuda")
    y = x @ x
    torch.cuda.synchronize()  # forces GPU sync
    return y

with profiler.profile(
    activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    for _ in range(3):
        sync_op()

# Save JSON trace (view in Chrome: chrome://tracing)
prof.export_chrome_trace("trace.json")
print("Trace saved to trace.json (open in chrome://tracing)")
