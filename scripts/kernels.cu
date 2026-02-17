// Low-level CUDA kernel examples
// Compile: nvcc -arch=sm_90 -O3 -o kernels kernels.cu
// Run: ./kernels

#include <cuda_runtime.h>
#include <stdio.h>
#include <mma.h>

using namespace nvcuda;

// =============================================================================
// Example 1: Naive Matrix Addition (baseline)
// =============================================================================
__global__ void naive_add(const float* A, const float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

// =============================================================================
// Example 2: Tiled GEMM with Shared Memory
// =============================================================================
#define TILE_DIM 16

__global__ void tiled_gemm(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    // Shared memory for tiles
    __shared__ float As[TILE_DIM][TILE_DIM];
    __shared__ float Bs[TILE_DIM][TILE_DIM];
    
    // Thread indices
    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int col = blockIdx.x * TILE_DIM + threadIdx.x;
    
    float sum = 0.0f;
    
    // Loop over K tiles
    for (int k = 0; k < K; k += TILE_DIM) {
        // Load A tile
        if (row < M && (k + threadIdx.x) < K) {
            As[threadIdx.y][threadIdx.x] = A[row * K + (k + threadIdx.x)];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        // Load B tile
        if (col < N && (k + threadIdx.y) < K) {
            Bs[threadIdx.y][threadIdx.x] = B[(k + threadIdx.y) * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial dot product
        #pragma unroll
        for (int i = 0; i < TILE_DIM; i++) {
            sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    // Store result
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// =============================================================================
// Example 3: Warp-Level Reduction (Sum)
// =============================================================================
__device__ float warp_reduce_sum(float val) {
    // Warp shuffle: each thread gets value from neighbor
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void warp_reduce_kernel(const float* input, float* output, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load and reduce within warp
    float val = (tid < N) ? input[tid] : 0.0f;
    val = warp_reduce_sum(val);
    
    // First thread writes block sum
    if (threadIdx.x == 0) {
        output[blockIdx.x] = val;
    }
}

// =============================================================================
// Example 4: Tensor Core MMA (WMMA) - FP16
// =============================================================================
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

__global__ void wmma_gemm_kernel(
    half* __restrict__ A,
    half* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    // Warp-level indices
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int warpN = blockIdx.y * blockDim.y + threadIdx.y;
    
    // WMMA fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    
    // Initialize accumulator
    wmma::fill_fragment(c_frag, 0.0f);
    
    // Loop over K
    for (int k = 0; k < K; k += WMMA_K) {
        // Load A
        if (warpM * WMMA_M + threadIdx.y < M && k + threadIdx.x < K) {
            wmma::load_matrix_sync(a_frag, A + (warpM * WMMA_M + threadIdx.y) * K + (k + threadIdx.x), K);
        }
        
        // Load B
        if (warpN * WMMA_N + threadIdx.x < N && k + threadIdx.y < K) {
            wmma::load_matrix_sync(b_frag, B + (k + threadIdx.y) * N + (warpN * WMMA_N + threadIdx.x), N);
        }
        
        // Multiply-Accumulate
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    
    // Store result
    if (warpM * WMMA_M + threadIdx.y < M && warpN * WMMA_N + threadIdx.x < N) {
        wmma::store_matrix_sync(
            C + (warpM * WMMA_M + threadIdx.y) * N + (warpN * WMMA_N + threadIdx.x),
            c_frag, N, wmma::mem_row_major
        );
    }
}

// =============================================================================
// Example 5: Atomic-Free Parallel Reduction
// =============================================================================
__global__ void parallel_reduce_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N
) {
    extern __shared__ float sdata[];
    
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    
    // Load into shared memory
    sdata[tid] = (idx < N) ? input[idx] : 0.0f;
    if (idx + blockDim.x < N) sdata[tid] += input[idx + blockDim.x];
    
    __syncthreads();
    
    // In-place reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && idx + s < N) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Write result for this block
    if (tid == 0) output[blockIdx.x] = sdata[0];
}

// =============================================================================
// Launcher functions (for testing)
// =============================================================================
void run_naive_add() {
    const int N = 1024 * 1024;
    float *h_A, *h_B, *h_C, *d_A, *d_B, *d_C;
    
    h_A = new float[N];
    h_B = new float[N];
    h_C = new float[N];
    
    cudaMalloc(&d_A, N * sizeof(float));
    cudaMalloc(&d_B, N * sizeof(float));
    cudaMalloc(&d_C, N * sizeof(float));
    
    // Initialize
    for (int i = 0; i < N; i++) {
        h_A[i] = i * 0.1f;
        h_B[i] = i * 0.2f;
    }
    
    cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice);
    
    // Launch
    int block = 256;
    int grid = (N + block - 1) / block;
    naive_add<<<grid, block>>>(d_A, d_B, d_C, N);
    
    cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    printf("naive_add: C[0] = %f (expected %f)\n", h_C[0], h_A[0] + h_B[0]);
    
    delete[] h_A; delete[] h_B; delete[] h_C;
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}

int main() {
    printf("CUDA Kernel Examples\n");
    printf("====================\n");
    
    run_naive_add();
    
    printf("Done!\n");
    return 0;
}
