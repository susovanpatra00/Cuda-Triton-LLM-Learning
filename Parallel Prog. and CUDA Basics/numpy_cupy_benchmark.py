import numpy as np
import cupy as cp
import time

N = 10000000
a = np.random.rand(N)
b = np.random.rand(N)

start = time.time()
c = a + b
end = time.time()
print("CPU time:", end - start)


a_gpu = cp.random.rand(N)
b_gpu = cp.random.rand(N)
start = time.time()

c_gpu = a_gpu + b_gpu
cp.cuda.Stream.null.synchronize()  # Wait for GPU
end = time.time()
print("GPU time:", end - start)
