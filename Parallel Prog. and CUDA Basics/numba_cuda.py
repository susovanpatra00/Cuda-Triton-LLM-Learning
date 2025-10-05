import numpy as np
from numba import cuda

# CUDA kernel
@cuda.jit
def vector_add(a, b, c):
    idx = cuda.grid(1)   # global thread ID
    if idx < a.size:
        c[idx] = a[idx] + b[idx]

# Host (CPU) code
N = 1000000
a = np.arange(N, dtype=np.float32)
b = np.arange(N, dtype=np.float32)
c = np.zeros(N, dtype=np.float32)

# Allocate GPU memory & copy data
d_a = cuda.to_device(a)
d_b = cuda.to_device(b)
d_c = cuda.device_array_like(c)

# Launch kernel (blocks, threads per block)
threads_per_block = 256
blocks = (N + threads_per_block - 1) // threads_per_block
vector_add[blocks, threads_per_block](d_a, d_b, d_c)

# Copy result back to CPU
c = d_c.copy_to_host()
print(c[:10])
