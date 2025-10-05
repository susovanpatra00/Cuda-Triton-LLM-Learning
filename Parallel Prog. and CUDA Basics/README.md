Alright, let’s go **deep and structured** on Module 1: Parallel Programming Fundamentals. I’ll go **theory → examples → practice** so you understand both the concepts and how to implement them.

---

## **1. Why CPUs Are Limited in Parallel Tasks**

### **a) The Power Wall**

* Modern CPUs can’t just keep increasing clock speed due to heat and power constraints.
* Higher frequency → more power consumption (∝ frequency³) → thermal limits.
* So, CPU designers started adding **more cores** instead of faster single cores.

### **b) Memory Bandwidth Bottleneck**

* CPUs can execute instructions very fast, but they still need to **fetch data from RAM**.
* Memory is slower than CPU, so for data-heavy operations, CPUs often **wait for data**.
* This is called being **memory-bound**, limiting speedup even if you have multiple cores.

**Takeaway:** CPUs are great for sequential tasks or tasks with modest parallelism, but for massively parallel workloads, they hit physical limits.

---

## **2. Amdahl’s Law**

Amdahl’s Law tells us the **maximum theoretical speedup** of a program if part of it can be parallelized.

$$
\text{Speedup}_{\text{max}} = \frac{1}{(1-P) + \frac{P}{N}}
$$

Where:

| Symbol | Meaning                                   |
| ------ | ----------------------------------------- |
| $P$    | Fraction of code that can run in parallel |
| $N$    | Number of cores                           |
| $1-P$  | Fraction that must run sequentially       |

### **Example**

Suppose:

* $P = 0.9$ (90% of code is parallel)
* $N = 4$ cores

$$
\text{Speedup}_{\text{max}} = \frac{1}{(1-0.9) + \frac{0.9}{4}} = \frac{1}{0.1 + 0.225} = \frac{1}{0.325} \approx 3.08
$$

Even with 4 cores, you **cannot get a 4× speedup** because the 10% sequential part is a bottleneck.

**Key insight:** Adding more cores gives **diminishing returns** if a significant part of the code is sequential.

---

## **3. Why GPUs Are Advantageous**

* GPUs have **thousands of smaller cores**, designed for tasks that can run in parallel (like vector or matrix operations).
* CPUs: 4–64 powerful cores (good for sequential code)
* GPUs: 1000+ simpler cores (good for massively parallel code)
* FLOPS (Floating Point Operations per Second) are **much higher** for GPUs in parallel workloads.

**Example:** Adding 10 million numbers

* CPU: each core adds a chunk → moderate speedup
* GPU: thousands of cores do the additions simultaneously → huge speedup

**Takeaway:** GPUs shine for **highly parallel, compute-heavy tasks** (linear algebra, image processing, ML).

---

## **4. Practical Exercises**

### **a) Compare CPU vs GPU in Python**

Install libraries if not present:

```bash
pip install numpy cupy
```

**CPU with NumPy:**

```python
import numpy as np
import time

N = 10_000_000
a = np.random.rand(N)
b = np.random.rand(N)

start = time.time()
c = a + b
end = time.time()
print("CPU time:", end - start)
```

**GPU with CuPy:**

```python
import cupy as cp
import time

a_gpu = cp.random.rand(N)
b_gpu = cp.random.rand(N)

start = time.time()
c_gpu = a_gpu + b_gpu
cp.cuda.Stream.null.synchronize()  # Wait for GPU
end = time.time()
print("GPU time:", end - start)
```

**Observation:** GPU will usually be faster for large arrays, but **for small arrays, CPU can be faster** because of GPU transfer overhead.

---

### **b) Break loops into parallel tasks**

Example: squaring numbers

**Sequential:**

```python
nums = list(range(10))
squared = [x**2 for x in nums]
```

**Parallel with `concurrent.futures`:**

```python
from concurrent.futures import ThreadPoolExecutor

def square(x):
    return x**2

with ThreadPoolExecutor() as executor:
    squared_parallel = list(executor.map(square, nums))

print(squared_parallel)
```

**Key Idea:** Each loop iteration is **independent**, so it can run in parallel. This is the foundation for **CPU or GPU parallelism**.

---

### **c) Tip for GPU Thinking**

* Ask: “Can this operation run on **many elements simultaneously**?”
* If yes → perfect for GPU (vectors, matrices, images).
* If no → likely stays on CPU (branch-heavy, sequential logic).

---

Let’s break this down **carefully and fully**.

---

## **1. What is Clock Speed?**

* The **clock speed** of a CPU is how fast the CPU’s internal clock ticks.
* Usually measured in **GHz (gigahertz)**: 1 GHz = 1 billion cycles per second.
* Each cycle is an opportunity for the CPU to do a basic operation (like add two numbers, read/write memory, etc.).

Think of the CPU as a **train station**:

* Each “tick” of the clock is a train arriving.
* Each train can carry passengers (instructions).
* Faster clock → more trains per second → more instructions processed per second.

---

## **2. Why Clock Speed Matters**

* **Higher clock speed = potentially more operations per second.**
* For a **single-core CPU**, a higher clock means tasks finish faster.

**Example:**

* CPU A: 2 GHz → 2 billion cycles/sec
* CPU B: 4 GHz → 4 billion cycles/sec

If all else is equal, CPU B can do roughly **twice the work in the same time**.

---

## **3. Why Higher Clock Isn’t Always Better**

1. **Heat & Power Limits (Power Wall)**

   * Increasing clock speed increases power consumption (roughly ∝ frequency³) → CPU gets hot → throttling needed.
2. **Memory Bottleneck**

   * CPU may be waiting for RAM. No matter how fast the clock is, it can’t execute instructions without data.
3. **Instruction Complexity**

   * Modern CPUs have complex instructions. Some take multiple cycles, so raw clock speed alone doesn’t tell the full story.

---

### **4. CPU vs GPU Perspective**

* CPU: **few powerful cores**, higher clock speeds (~3–5 GHz), good for sequential tasks.
* GPU: **thousands of simpler cores**, lower clock speeds (~1–2 GHz), but massive parallel throughput.

**Key:** For single-threaded performance → CPU clock matters.
For massively parallel workloads → core count and architecture (GPU) matter more.

---

## **1. Single-Core Speed vs Multi-Core**

* Clock speed tells you how fast **one core** executes instructions.
* If a task runs **entirely on one core**, higher GHz → faster execution.
* If a task can be **parallelized across multiple cores**, total performance depends on:

  1. Number of cores $N$
  2. How much of the task can be parallelized $P$ (Amdahl’s Law)
  3. Clock speed of each core

---

## **2. Combining Clock Speed with Amdahl’s Law**

Amdahl’s Law:

$$
\text{Speedup}_{\text{max}} = \frac{1}{(1-P) + \frac{P}{N}}
$$

Now, let’s add clock speed:

* Suppose CPU cores run at **f GHz**
* Sequential portion $1-P$ executes on **one core** at f GHz
* Parallel portion $P$ executes across N cores, each at f GHz

**Effective execution time:**

$$
T_\text{effective} = \frac{(1-P)}{f} + \frac{P}{N \cdot f}
$$

* Higher **f** → shorter sequential and parallel times
* Higher **N** → shorter parallel time only
* But **sequential part (1-P)** still dominates if it’s significant

---

### **Example**

* Task: 90% parallel ($P=0.9$), 4 cores ($N=4$)
* CPU A: 3 GHz
* CPU B: 4 GHz

**Sequential time on CPU A:** $(1-P)/f = 0.1 / 3 \approx 0.033$
**Parallel time on CPU A:** $P / (N \cdot f) = 0.9 / (4*3) = 0.075$
**Total effective time:** 0.033 + 0.075 = 0.108 (relative units)

**CPU B (4 GHz)**:
Sequential: 0.1 / 4 = 0.025
Parallel: 0.9 / (4*4) = 0.05625
Total: 0.08125

✅ **Observation:** Increasing GHz reduces total time, but **adding more cores reduces only the parallel portion**.
✅ **Observation:** Even infinite cores → sequential part is a hard limit.

---

## **3. Why You Can’t Just Increase Clock for Speedup**

* Sequential fraction $1-P$ dominates with many cores → diminishing returns.
* CPU clock helps, but physical **heat/power limits** cap it.
* True massive speedups need **parallel-friendly architectures** like GPUs.

**Summary:**

| Factor              | Helps What                                 |
| ------------------- | ------------------------------------------ |
| Clock speed (GHz)   | Single-threaded / sequential performance   |
| Number of cores (N) | Parallel fraction (P) speedup              |
| GPU                 | Massive parallel fraction (high P) speedup |

---





**Module 2: GPU and CUDA Basics** 
---

## **Theory**

### 1. CUDA Terminology

* **Host** → Your **CPU** and main memory (RAM). Runs normal sequential code.
* **Device** → Your **GPU** and its memory (VRAM). Runs massively parallel code.
* **Kernel** → A GPU function you write (special CUDA code) that runs on many threads in parallel.
* **Thread** → Smallest execution unit on the GPU (like a CPU thread, but much lighter).
* **Block** → A group of threads.
* **Grid** → A group of blocks.

So when you launch a kernel, you decide:
**How many threads per block?** and **How many blocks in the grid?**
That determines how much parallel work runs.

---

### 2. Heterogeneous Computing

* CPU → good at **sequential tasks** (logic, branching, orchestration).
* GPU → good at **parallel tasks** (same computation repeated many times, like matrix multiply).
* You use both together: CPU organizes the problem, GPU executes the heavy parallel math.

---

### 3. Embarrassingly Parallel Tasks

* A task is "embarrassingly parallel" if each piece of work is **independent**.
  Examples:
* Vector addition (each element sum doesn’t depend on others).
* Image filtering pixel-by-pixel.
* Monte Carlo simulations (each trial independent).

---

## **Practice**

### Step 2: Write GPU Code in Python

We’ll use **CuPy** (NumPy-like but on GPU).

```python
import cupy as cp

# Vector size
N = 1_000_000  

# Create vectors on GPU
a = cp.arange(N, dtype=cp.float32)
b = cp.arange(N, dtype=cp.float32)

# Vector addition on GPU
c = a + b

print(c[:10])   # first 10 results
```

That single line `c = a + b` runs on GPU, very fast.

---

### Step 3: Custom Kernel with Numba

If you want to see **how CUDA kernels look**, try Numba:

```python
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
```

Here:

* `cuda.grid(1)` → gets unique thread ID.
* `vector_add[blocks, threads_per_block](...)` → launches GPU kernel with that many threads.

---

### Step 4: Find Independent Loops in Python

Take any loop in your Python code:

Example:

```python
result = []
for i in range(len(a)):
    result.append(a[i] ** 2 + 3 * b[i])
```

Each iteration only depends on `i`. No cross-dependency → **embarrassingly parallel** → good for GPU.

But:

```python
s = 0
for i in range(len(a)):
    s += a[i]   # depends on previous iteration
```

This one is **not independent** (reduction, needs special handling).

---


 **Module 3: CUDA C Programming**

---

## **Theory**

### 1. Kernel types

CUDA uses special qualifiers to mark where functions run and who can call them:

* **`__global__`**
  *Executed on GPU, called from CPU (host)*.
  Example: a kernel function.

* **`__device__`**
  *Executed on GPU, called from GPU (other device functions)*.
  Example: helper functions for kernels.

* **`__host__`**
  *Executed on CPU, called from CPU*.
  This is default in C/C++.

> You can also combine `__host__ __device__` to make a function usable on both CPU and GPU.

---

### 2. Thread indexing

CUDA uses a grid–block–thread hierarchy:

* **`threadIdx`** → thread’s index inside a block (0 … blockDim.x − 1).
* **`blockIdx`** → block’s index inside the grid (0 … gridDim.x − 1).
* **`blockDim`** → number of threads per block.

So the **global thread index** is:

```c
int i = blockIdx.x * blockDim.x + threadIdx.x;
```

---

### 3. Memory management

CUDA has **separate memory spaces** for host (CPU) and device (GPU). You need explicit memory copies.

* **`cudaMalloc((void**)&ptr, size)`** → allocate GPU memory.
* **`cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice)`** → copy data from host to device.
* **`cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost)`** → copy back results.
* **`cudaFree(ptr)`** → free GPU memory.

---

### 4. Kernel launching

You launch GPU functions (kernels) with special syntax:

```c
kernel<<<numBlocks, numThreads>>>(args...);
```

* `numBlocks` → how many blocks in the grid.
* `numThreads` → how many threads per block.

Example:

```c
vecAdd<<<(N+255)/256, 256>>>(A, B, C, N);
```

This launches enough threads to cover `N` elements with 256 threads per block.

---

## **Practice: Vector Addition**

We’ll write a CUDA program that adds two vectors `C = A + B`.

### Step 1: Kernel

```c
__global__ void vectorAdd(float *A, float *B, float *C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {   // boundary check
        C[i] = A[i] + B[i];
    }
}
```

### Step 2: Host code

```c
#include <stdio.h>
#include <cuda_runtime.h>

#define N 1000   // vector size

int main() {
    int size = N * sizeof(float);
    float *h_A, *h_B, *h_C;         // host arrays
    float *d_A, *d_B, *d_C;         // device arrays

    // Allocate host memory
    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    h_C = (float*)malloc(size);

    // Initialize host arrays
    for (int i = 0; i < N; i++) {
        h_A[i] = i * 1.0f;
        h_B[i] = i * 2.0f;
    }

    // Allocate device memory
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Copy data host → device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Copy result device → host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Verify
    for (int i = 0; i < 10; i++) {
        printf("%f + %f = %f\n", h_A[i], h_B[i], h_C[i]);
    }

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
```

---

### **How to compile & run**

If you have CUDA installed:

```bash
nvcc vector_add.cu -o vector_add
./vector_add
```

Output should show first few additions:

```
0.000000 + 0.000000 = 0.000000
1.000000 + 2.000000 = 3.000000
2.000000 + 4.000000 = 6.000000
...
```

---



Great — let’s build intuition about **how threads are organized on the GPU** and go over **the common CUDA terminologies** so the mental picture is crystal clear.

---

## **Behind the scenes of a kernel launch**

When you write:

```c
vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
```

you’re telling CUDA:

* **Launch `blocksPerGrid` blocks**
* **Each block has `threadsPerBlock` threads**

So the **total number of threads = blocksPerGrid × threadsPerBlock**.

---

### Example:

Let’s say:

```c
int threadsPerBlock = 256;
int blocksPerGrid = 4;
```

Then we launch **4 blocks × 256 threads = 1024 threads**.

Each thread computes **one index `i`**:

```c
i = blockIdx.x * blockDim.x + threadIdx.x;
```

* `threadIdx.x` → local index of thread inside a block (0–255).
* `blockIdx.x` → index of the block in the grid (0–3).
* `blockDim.x` → total threads per block (256).

So for block 2 (i.e., `blockIdx.x = 2`) and thread 5 (`threadIdx.x = 5`):

```c
i = 2 * 256 + 5 = 517
```

That thread will compute `C[517] = A[517] + B[517]`.

---

### Visualization (1D grid of 4 blocks, each with 8 threads):

```
Grid
 └── Block 0: threads [0,1,2,3,4,5,6,7]
 └── Block 1: threads [8,9,10,11,12,13,14,15]
 └── Block 2: threads [16,17,18,19,20,21,22,23]
 └── Block 3: threads [24,25,26,27,28,29,30,31]
```

Each thread is responsible for one element of the array.

For 2D or 3D problems (like image processing), you can also use `threadIdx.y`, `blockIdx.y`, etc., to map threads to rows and columns.

---

## **Common CUDA Terminologies**

1. **Host**

   * Refers to the **CPU** and its memory (RAM).
   * Runs the normal C/C++ code, allocates memory with `malloc()`, and controls the GPU by launching kernels.

2. **Device**

   * Refers to the **GPU** and its global memory (VRAM).
   * Executes CUDA kernels in parallel.

3. **Kernel**

   * A GPU function, marked with `__global__`, launched by the host.
   * Runs on thousands of lightweight GPU threads simultaneously.

4. **Thread**

   * The smallest execution unit on the GPU.
   * Each thread computes one piece of data (e.g., one element in vector addition).

5. **Block**

   * A group of threads (1D, 2D, or 3D).
   * Identified by `blockIdx`.
   * Size given by `blockDim`.

6. **Grid**

   * A collection of blocks.
   * Identified by `gridDim`.
   * Can also be 1D, 2D, or 3D.

7. **Global Memory**

   * Large memory on GPU (VRAM).
   * Accessible by all threads but has high latency.
   * You allocate with `cudaMalloc`.

8. **Shared Memory**

   * Fast, small memory shared by all threads **within a block**.
   * Useful for communication and avoiding repeated global memory access.

9. **Local Memory**

   * Memory private to each thread (like stack variables).

10. **Registers**

    * The fastest memory, private to each thread.
    * GPU uses registers heavily for computations.

---

## **Putting it all together**

* **Host code** (CPU): Allocates memory, copies data, launches kernels.
* **Device code** (GPU): Runs kernels, each thread computes one part of the result.
* **Hierarchy**:

  ```
  Grid → Blocks → Threads
  ```

---



