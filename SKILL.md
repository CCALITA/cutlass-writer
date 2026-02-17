---
name: cuda-kernel-coder
description: AI skill for writing high-performance CUDA kernels. When user asks to implement GPU compute, optimize CUDA code, or write low-level kernel - use this skill.
tags: [cuda, gpu, kernel, performance, nvidia]
triggers:
  - "write cuda kernel"
  - "implement gpu"
  - "optimize cuda"
  - "cuda performance"
  - "gpu kernel"
  - "write gemm"
  - "tensor core"
  - "cuda optimization"
version: 4.0.0
source: https://github.com/NVIDIA/cutlass
author: Skill Factory
---

# CUDA Kernel Coder

**Task: Write a high-performance CUDA kernel for the user's problem.**

---

## Step 1: Understand the Workload

Ask the user or infer:

1. **Matrix dimensions** (M, N, K)
2. **Data type** (float32, float16, bf16, int8)
3. **Operation type** (GEMM, reduction, custom)
4. **Target architecture** (Ampere 80, Hopper 90, Blackwell 100)
5. **Performance target** (TFLOPS, latency)

---

## Step 2: Choose Implementation

### Decision Tree

```
Is it matrix multiplication (GEMM)?
├── YES → Use Tensor Core (WMMA) if FP16/BF16
│   ├── M,N > 512 → Use Tiled GEMM with SMEM
│   └── Small M,N → Warp-level MMA
├── NO
    ├── Is it reduction? → Warp shuffle + block reduce
    ├── Is it element-wise? → Naive kernel + fuse
    └── Is it custom? → Use CuTe / CUTLASS
```

---

## Step 3: Quick Templates

### Template 1: Basic Element-wise

```cpp
// Usage: Vector add, scale, element-wise ops
__global__ void element_wise_kernel(
    const T* __restrict__ A,
    const T* __restrict__ B,
    T* __restrict__ C,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = op(A[idx], B[idx]);  // Replace op
    }
}
// Launch: element_wise_kernel<<<(N+255)/256, 256>>>(d_A, d_B, d_C, N);
```

### Template 2: Tiled GEMM (Shared Memory)

```cpp
// Usage: Medium-large GEMM, any precision
#define TILE 16

__global__ void tiled_gemm(
    const float* A, const float* B, float* C,
    int M, int N, int K
) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];
    
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.0f;
    
    for (int k = 0; k < K; k += TILE) {
        As[threadIdx.y][threadIdx.x] = (row < M && k + threadIdx.x < K) ? A[row*K + k + threadIdx.x] : 0;
        Bs[threadIdx.y][threadIdx.x] = (col < N && k + threadIdx.y < K) ? B[(k+threadIdx.y)*N + col] : 0;
        __syncthreads();
        #pragma unroll
        for (int i = 0; i < TILE; i++) sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < N) C[row*N + col] = sum;
}
// Launch: dim3 grid((N+15)/16, (M+15)/16), dim3(16,16)
```

### Template 3: Tensor Core MMA (FP16/BF16)

```cpp
// Usage: High-performance GEMM on Ampere/Hopper/Blackwell
// Note: Requires -arch=sm_80+

#define WMMA_M 16, WMMA_N 16, WMMA_K 16

__global__ void wmma_gemm(half* A, half* B, float* C, int M, int N, int K) {
    int warpM = (blockIdx.x * 32 + threadIdx.x) / 32;
    int warpN = blockIdx.y * 16 + threadIdx.y;
    
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    
    wmma::fill_fragment(c_frag, 0.0f);
    
    for (int k = 0; k < K; k += WMMA_K) {
        if (warpM*16 + threadIdx.y < M && k + threadIdx.x < K)
            wmma::load_matrix_sync(a_frag, A + (warpM*16 + threadIdx.y)*K + k, K);
        if (warpN*16 + threadIdx.x < N && k + threadIdx.y < K)  
            wmma::load_matrix_sync(b_frag, B + (k + threadIdx.y)*N + warpN*16, N);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    if (warpM*16+threadIdx.y < M && warpN*16+threadIdx.x < N)
        wmma::store_matrix_sync(C + (warpM*16+threadIdx.y)*N + warpN*16, c_frag, N, wmma::mem_row_major);
}
// Launch: dim3 grid((M+15)/16, (N+15)/16), dim3(32,16)
```

### Template 4: Warp Reduction

```cpp
// Usage: Sum, max, min across warps
__device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void reduce_kernel(const float* input, float* output, int N) {
    float val = (blockIdx.x * blockDim.x + threadIdx.x < N) 
                ? input[blockIdx.x * blockDim.x + threadIdx.x] : 0;
    val = warp_reduce_sum(val);
    if (threadIdx.x == 0) atomicAdd(&output[blockIdx.x], val);
}
```

---

## Step 4: Common Patterns

### Pattern: Choose Tile Size

| Problem Size | Tile | Use |
|-------------|------|-----|
| M,N < 128 | 8 | Small, avoid waste |
| M,N 128-1024 | 16 | Balanced |
| M,N > 1024 | 32 | Large, more parallelism |

### Pattern: Data Type Selection

| Need | Use |
|------|-----|
| Highest precision | float32 |
| Fast, good accuracy | float16 + FP32 accumulator |
| ML training | bf16 |
| ML inference | int8 / fp8 |
| Massive throughput | fp8 (Hopper+) |

### Pattern: Launch Config

```cpp
// Standard config
int block = 256;  // 8 warps
int grid = (N + block - 1) / block;

// For Tensor Core
dim3 block(32, 16);  // 1 warp per dim
dim3 grid((M+15)/16, (N+15)/16);
```

---

## Step 5: Compilation

```bash
# Basic
nvcc -arch=sm_80 -O3 kernel.cu -o kernel

# Tensor Core + Fast Math
nvcc -arch=sm_90 -O3 --use_fast_math -Xptxas -v kernel.cu -o kernel

# With debug symbols
nvcc -arch=sm_90 -g -G -O3 kernel.cu -o kernel
```

---

## Step 6: Common Fixes

| Error | Fix |
|-------|-----|
| Wrong results | Add FP32 accumulator: `float acc = 0; acc += a*b;` |
| Slow | Check tile size, enable Tensor Core |
| OOM | Reduce tile to 16 or 8 |
| Bank conflict | Add padding: `shared[threadIdx.y][threadIdx.x+PAD]` |
| Divergent warp | Avoid `if` inside warp, use `__shfl` |

---

## Step 7: Verify & Profile

```bash
# Correctness
./kernel

# Profile
nsys profile --trace=cuda -o profile ./kernel
nsight-compute ./profile.qdrep

# Occupancy
nvcc -Xptxas -v -arch=sm_90 kernel.cu
```

---

## Quick Reference

```
┌─────────────────────────────────────────────┐
│           Decision Cheat Sheet              │
├─────────────────────────────────────────────┤
│ GEMM + FP16 → wmma_gemm template           │
│ GEMM + FP32 → tiled_gemm template          │
│ Element-wise → element_wise template        │
│ Reduce/Sum → warp_reduce_sum + kernel      │
│ Custom op → Use CUTLASS CuTe               │
│ Don't know → Ask for M,N,K,dtype,arch     │
└─────────────────────────────────────────────┘
```

---

## Tools to Use

- `nvcc` - Compile
- `nsys` - Profile
- `nsight-compute` - Analyze
- `cuda-gdb` - Debug
- `nvvp` - Visual profiler
