# ===============================
# CPU vs GPU Timings + Amdahl Law
# ===============================

import numpy as np
import cupy as cp
import time
import matplotlib.pyplot as plt

# -------------------------------
# Experiment: Vector Addition
# -------------------------------
N = 10000000
print("Creating arrays...")
a_cpu = np.random.rand(N)
b_cpu = np.random.rand(N)
a_gpu = cp.random.rand(N)
b_gpu = cp.random.rand(N)

# CPU timing
start = time.time()
c_cpu = a_cpu + b_cpu
end = time.time()
cpu_time = end - start
print(f"CPU time: {cpu_time:.5f} s")

# GPU timing
start = time.time()
c_gpu = a_gpu + b_gpu
cp.cuda.Stream.null.synchronize()  # Wait for GPU completion
end = time.time()
gpu_time = end - start
print(f"GPU time: {gpu_time:.5f} s")

# -------------------------------
# Amdahl's Law Speedup Simulation
# -------------------------------
def amdahl_speedup(P, N):
    return 1 / ((1-P) + P/N)

cores = np.arange(1, 33)
parallel_fractions = [0.5, 0.75, 0.9, 0.99]

plt.figure(figsize=(10,6))
for P in parallel_fractions:
    speedup = [amdahl_speedup(P, n) for n in cores]
    plt.plot(cores, speedup, label=f"P={P}")

plt.xlabel("Number of Cores")
plt.ylabel("Speedup")
plt.title("Amdahl's Law Speedup vs Cores")
plt.grid(True)
plt.legend()
plt.show()

# -------------------------------
# CPU vs GPU Visual Comparison
# -------------------------------
plt.figure(figsize=(8,5))
plt.bar(["CPU", "GPU"], [cpu_time, gpu_time], color=['skyblue','orange'])
plt.ylabel("Time (seconds)")
plt.title("CPU vs GPU Vector Addition")
plt.show()
