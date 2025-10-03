import torch
import torch.nn as nn
import torch.profiler as profiler

# Simple Transformer-like block (attention + FFN)
class MiniBlock(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads=8, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim*4),
            nn.ReLU(),
            nn.Linear(dim*4, dim)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        ff_out = self.ff(x)
        return self.norm2(x + ff_out)

device = "cuda"
model = MiniBlock().to(device)
x = torch.randn(16, 128, 512, device=device)  # (batch, seq_len, dim)

with profiler.profile(
    activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
    schedule=profiler.schedule(wait=1, warmup=1, active=3),
    on_trace_ready=profiler.tensorboard_trace_handler("./log"),
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    for step in range(5):
        with profiler.record_function("training_step"):
            out = model(x)
            loss = out.mean()
            loss.backward()
        prof.step()  # <-- IMPORTANT

# Console view
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
