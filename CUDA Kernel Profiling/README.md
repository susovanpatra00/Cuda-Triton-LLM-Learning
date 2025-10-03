# 🔥 CUDA Kernel Profiling in PyTorch

A comprehensive guide to understanding, profiling, and optimizing CUDA kernels in PyTorch, with practical examples and Triton integration.

## Table of Contents

1. [Understanding CUDA Kernels](#understanding-cuda-kernels)
2. [Why Profile CUDA Kernels?](#why-profile-cuda-kernels)
3. [Basic Profiling with PyTorch](#basic-profiling-with-pytorch)
4. [Advanced Profiling Techniques](#advanced-profiling-techniques)
5. [Analyzing Profiling Results](#analyzing-profiling-results)
6. [Common Performance Issues](#common-performance-issues)
7. [Advanced Profiling Tools](#advanced-profiling-tools)
8. [LLM-Specific Considerations](#llm-specific-considerations)
9. [Triton Integration](#triton-integration)
10. [Mastery Checklist](#mastery-checklist)

---

## Understanding CUDA Kernels

### What are CUDA kernels?

* A **CUDA kernel** is a GPU function launched in parallel across thousands of threads
* Every PyTorch operation that uses the GPU (`matmul`, `conv2d`, etc.) executes as one or more CUDA kernels
* Example: `y = x @ w` on GPU internally calls a matrix multiplication kernel implemented in CUDA


---

### CUDA Execution Model

Understanding how PyTorch interacts with CUDA is key to effective profiling:

* **CPU schedules, GPU executes:** Every PyTorch operation starts on the **CPU**. The CPU decides which backend to use (CPU kernel or CUDA kernel) based on the device of the input tensors, checks shapes, types, and broadcasting rules, then **launches the corresponding CUDA kernel** on the GPU.
* **Why not directly on GPU:** The GPU cannot execute Python code, interpret tensor shapes, or handle complex scheduling logic. It only runs compiled kernels. The CPU acts as the **controller**, orchestrating which kernels to launch, which streams to use, and handling fallback or mixed-device cases.
* **Asynchronous execution:** Kernel launches on the GPU are **non-blocking** for the CPU. This allows the CPU to continue scheduling more operations while the GPU executes in parallel.
* **GPU as a worker:** The GPU simply executes the kernels enqueued by the CPU across thousands of threads.
* **Synchronization points:** The CPU only waits for GPU results when necessary (e.g., `.cpu()`, `.item()`, or explicit `torch.cuda.synchronize()`). Until then, GPU operations run independently.
* **Multiple CUDA streams:** PyTorch can execute kernels on multiple streams in parallel (default is `torch.cuda.default_stream()`), enabling overlapping computation and memory transfers.
* **Profiling perspective:** Tools like `torch.profiler` show **CPU scheduling time**, **CUDA execution time**, and **synchronization overhead**, helping identify bottlenecks and optimize performance.

**Analogy:** The CPU is the **chef**, orchestrating the recipe and delegating tasks, while the GPU is the **kitchen staff**, executing instructions in parallel at high speed.


---

## Why Profile CUDA Kernels?

Training/inference on GPU can be slow for various reasons:

* **Kernel bottlenecks**: Certain kernels taking too long
* **Memory transfers**: CPU ↔ GPU data movement bottlenecks
* **Launch overhead**: Too many small kernels being launched
* **Underutilization**: Batch size too small to fully utilize GPU
* **Synchronization**: Unnecessary CPU-GPU synchronization points

Profiling helps you **identify where GPU time is spent** so you can optimize effectively.

---

## Basic Profiling with PyTorch

### Simple Profiling Example

```python
import torch
import torch.profiler as profiler

x = torch.randn(1000, 1000, device="cuda")
w = torch.randn(1000, 1000, device="cuda")

def training_step():
    y = x @ w  # matrix multiplication (runs on GPU)
    return y

with profiler.profile(
    activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
    record_shapes=True
) as prof:
    with profiler.record_function("train_step"):
        training_step()

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

### Understanding the Output

The profiler output shows a table of operations:

```
-------------------------------  ------------  ------------  ------------
Name                             CPU time (us) CUDA time (us)  # of Calls
-------------------------------  ------------  ------------  ------------
matmul                           200.000       1500.000      1
empty_like                       5.000         0.000         1
to                               50.000        20.000        1
-------------------------------  ------------  ------------  ------------
```

This tells you:
* `matmul` CUDA kernel took **1.5 ms on GPU**
* Some operations are CPU-only (like tensor creation)
* Sort by CUDA time to find **bottlenecks**

---

## Advanced Profiling Techniques

### Comprehensive Profiling Setup

```python
import torch
import torch.profiler as profiler

def model_step(x, w):
    return x @ w + torch.relu(x)

x = torch.randn(4096, 4096, device="cuda")
w = torch.randn(4096, 4096, device="cuda")

with profiler.profile(
    activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
    schedule=profiler.schedule(wait=1, warmup=1, active=2),
    on_trace_ready=profiler.tensorboard_trace_handler("./log"),
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    for step in range(5):
        with profiler.record_function("training_step"):
            y = model_step(x, w)
        prof.step()

torch.cuda.synchronize()
```

**Key Parameters:**
* `schedule(...)`: Controls warmup/active phases (avoids startup noise)
* `on_trace_ready(...)`: Saves results for visualization
* `record_shapes=True`: Logs tensor shapes for each kernel
* `profile_memory=True`: Logs GPU memory allocation
* `with_stack=True`: Shows which Python line launched the kernel

---

## Analyzing Profiling Results

### Console Summary

```python
print(prof.key_averages().table(sort_by="cuda_time_total"))
```

### TensorBoard Timeline

```bash
tensorboard --logdir=./log
```

Navigate to **Profiler tab** to see:
* Timeline of CPU vs CUDA kernels
* GPU idle time gaps
* Operation breakdown
* Memory usage patterns

### Chrome Trace Viewer

```python
prof.export_chrome_trace("trace.json")
# Open trace.json in chrome://tracing
```

---

## Common Performance Issues

### 1. Kernel Bottlenecks
* **Issue**: Specific kernels consuming most CUDA time
* **Solution**: Use efficient fused kernels (e.g., FlashAttention vs regular attention)
* **Check**: Which kernels take most time? (`matmul`, `softmax`, `layer_norm`)

### 2. Launch Overhead
* **Issue**: Too many small kernels → CPU bottleneck
* **Solution**: Use fused operations
* **Example**: Fused LayerNorm vs multiple separate kernels

### 3. Synchronization Points
* **Issue**: Operations forcing GPU-CPU synchronization
* **Culprits**: `.item()`, `.cpu()`, `.numpy()`
* **Solution**: Avoid synchronization in training loops

### 4. Memory Bottlenecks
* **Issue**: Excessive CPU ↔ GPU data transfers
* **Solution**: Keep data on GPU, avoid `tensor.to("cuda")` in loops
* **Check**: Memory bandwidth utilization

### 5. GPU Underutilization
* **Issue**: GPU idle time in timeline
* **Causes**: Small batch size, lightweight model
* **Solutions**: Increase batch size, mixed precision, fused ops

### 6. Stream Usage
* **Issue**: All kernels in single stream
* **Solution**: Use multiple streams for compute/communication overlap
* **Important**: Multi-GPU training scenarios

---

## Advanced Profiling Tools

### NVIDIA Nsight Systems

Shows detailed CPU-GPU timeline:

```bash
nsys profile -o out_report python train.py
```

### NVIDIA Nsight Compute

Provides kernel-level metrics (FLOPs, memory throughput, warp efficiency):

```bash
ncu --target-processes all python train.py
```

These tools help determine if kernels are compute-bound or memory-bound.

---

## LLM-Specific Considerations

When training/serving large language models, focus on:

### Critical Kernels
* **Attention kernels**: Use FlashAttention, monitor GPU utilization
* **Matrix multiplications (GEMMs)**: Ensure Tensor Core usage (BF16/FP16)
* **LayerNorm/activations**: Fuse when possible

### System-Level Issues
* **Data pipeline**: CPU dataloading can stall GPU
* **Distributed training**: NCCL kernels (`all_reduce`, `all_gather`) often bottleneck
* **Memory management**: Profile memory allocation patterns

---

## Triton 

### Installation

```bash
pip install triton
```

Verify CUDA availability:
```python
import torch
print(torch.cuda.is_available())  # Should be True
```

### Writing a Simple Triton Kernel

```python
import torch
import triton
import triton.language as tl

# Triton kernel to compute y = 2*x + 3
@triton.jit
def fused_kernel(X_ptr, Y_ptr, N: tl.constexpr):
    pid = tl.program_id(0)  # Unique block ID
    # Each block processes 1024 elements
    offset = pid * 1024 + tl.arange(0, 1024)
    mask = offset < N  # Avoid out-of-bounds access
    x = tl.load(X_ptr + offset, mask=mask)  # Load input
    y = 2 * x + 3                            # Fused computation
    tl.store(Y_ptr + offset, y, mask=mask)   # Store output
```

**Key Concepts:**
* `tl.program_id(0)`: Block index (like `blockIdx` in CUDA C++)
* `tl.arange(0, 1024)`: Thread indices within block
* `mask`: Prevents out-of-bounds memory access
* `tl.load`/`tl.store`: Safe GPU memory operations

### Running the Triton Kernel

```python
# Create tensors
x = torch.randn(5000, device="cuda")
y = torch.empty_like(x)

# Launch kernel
grid = ((x.numel() + 1024 - 1) // 1024,)  # Number of blocks
fused_kernel[grid](x, y, x.numel())

print(y[:10])  # Check results
```

### Profiling Triton Kernels

```python
import torch.profiler as profiler

with profiler.profile(
    activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True
) as prof:
    fused_kernel[grid](x, y, x.numel())
    torch.cuda.synchronize()  # Ensure completion

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```
---

## Summary

**CUDA kernel profiling in PyTorch** means systematically analyzing which GPU operations run, their execution times, and identifying performance bottlenecks. This is essential for:

* **Speeding up training**: Identify and optimize slow kernels
* **Debugging performance**: Find synchronization issues and memory bottlenecks  
* **Resource optimization**: Maximize GPU utilization
* **Custom kernel development**: Profile and optimize Triton kernels

Master these techniques to build high-performance deep learning systems, especially for large language models where every millisecond counts.
