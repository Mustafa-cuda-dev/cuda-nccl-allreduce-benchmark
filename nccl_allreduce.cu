#include <cuda_runtime.h>
#include <nccl.h>
#include <cuda_fp16.h>
#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <string>
#include <type_traits>
#include <cstdlib>

// ==============================================================================
// ERROR HANDLING MACROS
// ==============================================================================
#define CUDA_CHECK(cmd) do { \
    cudaError_t err = (cmd); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "[FATAL CUDA ERROR] %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

#define NCCL_CHECK(cmd) do { \
    ncclResult_t res = (cmd); \
    if (res != ncclSuccess) { \
        fprintf(stderr, "[FATAL NCCL ERROR] %s:%d: %s\n", \
                __FILE__, __LINE__, ncclGetErrorString(res)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// ==============================================================================
// TYPE TRAITS FOR DATA CHARACTERIZATION
// ==============================================================================
enum BenchDataType {
    TYPE_FP32 = 0,
    TYPE_FP16 = 1,
    TYPE_INT8 = 2
};

template <typename T>
struct TypeTraits;

template <>
struct TypeTraits<float> {
    static const char* name() { return "FP32"; }
    static ncclDataType_t nccl_type() { return ncclFloat; }
    static BenchDataType enum_type() { return TYPE_FP32; }
};

template <>
struct TypeTraits<half> {
    static const char* name() { return "FP16"; }
    static ncclDataType_t nccl_type() { return ncclHalf; }
    static BenchDataType enum_type() { return TYPE_FP16; }
};

template <>
struct TypeTraits<int8_t> {
    static const char* name() { return "INT8"; }
    static ncclDataType_t nccl_type() { return ncclInt8; }
    static BenchDataType enum_type() { return TYPE_INT8; }
};

// ==============================================================================
// ARITHMETIC ENFORCERS FOR BASELINE SUMMATION
// ==============================================================================
template <typename T>
struct DeviceAdd {
    __device__ static __forceinline__ T apply(T a, T b) {
        return a + b;
    }
};

template <>
struct DeviceAdd<half> {
    __device__ static __forceinline__ half apply(half a, half b) {
#if __CUDA_ARCH__ >= 530
        return __hadd(a, b);
#else
        return __float2half(__half2float(a) + __half2float(b));
#endif
    }
};

// ==============================================================================
// CUSTOM REENTRANT BARRIER FOR MULTI-THREAD RUNTIME
// ==============================================================================
struct ThreadBarrier {
    std::mutex mtx;
    std::condition_variable cv;
    int target;
    int count = 0;
    int generation = 0;

    ThreadBarrier(int t) : target(t) {}

    void wait() {
        std::unique_lock<std::mutex> lock(mtx);
        int gen = generation;
        count++;
        if (count == target) {
            count = 0;
            generation++;
            cv.notify_all();
        } else {
            cv.wait(lock, [this, gen] { return gen != generation; });
        }
    }
};

// ==============================================================================
// CUDA KERNELS: INITIALIZATION & BASELINE REDUCTION
// ==============================================================================
template <typename T>
__device__ __forceinline__ T convert_value(float val);

template <>
__device__ __forceinline__ float convert_value<float>(float val) { return val; }

template <>
__device__ __forceinline__ half convert_value<half>(float val) { return __float2half(val); }

template <>
__device__ __forceinline__ int8_t convert_value<int8_t>(float val) { return (int8_t)val; }

template <typename T>
__global__ void initKernel(T* data, size_t elements, int rank) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)blockDim.x * gridDim.x;
    for (size_t i = idx; i < elements; i += stride) {
        // Deterministic sequence: avoids floating-point overflow for large arrays
        float val = (float)((rank + 1) * ((i % 7) + 1));
        data[i] = convert_value<T>(val);
    }
}

// Optimized Grid-Stride Reduction Kernel matching Framework 7 constraint parameters
template <typename T>
__global__ void __launch_bounds__(256, 4) vectorizedAddKernel(T* __restrict__ out, const T* __restrict__ in, size_t elements) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)blockDim.x * gridDim.x;
    
    #pragma unroll 4
    for (size_t i = idx; i < elements; i += stride) {
        out[i] = DeviceAdd<T>::apply(out[i], in[i]);
    }
}

// ==============================================================================
// VERIFICATION ENFORCER (SAMPLED FOR OUT-OF-MEMORY PREVENTATIVE RUNTIME)
// ==============================================================================
template <typename T>
bool verify_results(const T* host_recv, size_t elements, int num_gpus) {
    std::vector<size_t> indices;
    if (elements <= 3000) {
        for (size_t i = 0; i < elements; ++i) indices.push_back(i);
    } else {
        // Sample boundaries to prevent performance stalls and save host memory bandwidth
        for (size_t i = 0; i < 1000; ++i) indices.push_back(i);
        for (size_t i = elements / 2 - 500; i < elements / 2 + 500; ++i) indices.push_back(i);
        for (size_t i = elements - 1000; i < elements; ++i) indices.push_back(i);
    }

    for (size_t idx : indices) {
        double expected = 0.0;
        for (int rank = 0; rank < num_gpus; ++rank) {
            expected += (double)((rank + 1) * ((idx % 7) + 1));
        }

        double actual = 0.0;
        if (std::is_same<T, float>::value) {
            actual = (double)(*((const float*)host_recv + idx));
        } else if (std::is_same<T, half>::value) {
            actual = (double)__half2float(*((const half*)host_recv + idx));
        } else if (std::is_same<T, int8_t>::value) {
            actual = (double)(*((const int8_t*)host_recv + idx));
        }

        double diff = std::abs(expected - actual);
        double tol = 1e-4;
        if (std::is_same<T, half>::value) {
            tol = 1e-1; // FP16 cumulative addition tolerance boundary
        } else if (std::is_same<T, int8_t>::value) {
            tol = 0.0;  // Integer exact mapping constraint
        }

        if (diff > tol) {
            printf("\n[ERROR] Core Verification Failure at index %zu: expected %.2f, got %.2f (diff: %.5f)\n", 
                   idx, expected, actual, diff);
            return false;
        }
    }
    return true;
}

// ==============================================================================
// SINGLE-GPU BASELINE DEFINITION
// ==============================================================================
template <typename T>
double run_single_gpu_baseline(size_t elements, int num_gpus, int bench_iters, int warmup_iters) {
    CUDA_CHECK(cudaSetDevice(0));
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    size_t bytes = elements * sizeof(T);
    
    // Dynamic SM topology calculation to prevent multi-device wave tail-effects
    int num_sm = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, 0));
    int grid_size = num_sm * 4;
    int threads_per_block = 256;
    
    std::vector<T*> d_inputs(num_gpus);
    for (int i = 0; i < num_gpus; ++i) {
        CUDA_CHECK(cudaMalloc(&d_inputs[i], bytes));
        initKernel<T><<<grid_size, threads_per_block, 0, stream>>>(d_inputs[i], elements, i);
        CUDA_CHECK(cudaGetLastError());
    }
    T* d_output;
    CUDA_CHECK(cudaMalloc(&d_output, bytes));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Warmup
    for (int w = 0; w < warmup_iters; ++w) {
        CUDA_CHECK(cudaMemcpyAsync(d_output, d_inputs[0], bytes, cudaMemcpyDeviceToDevice, stream));
        for (int i = 1; i < num_gpus; ++i) {
            vectorizedAddKernel<T><<<grid_size, threads_per_block, 0, stream>>>(d_output, d_inputs[i], elements);
            CUDA_CHECK(cudaGetLastError());
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start, stream));
    for (int b = 0; b < bench_iters; ++b) {
        CUDA_CHECK(cudaMemcpyAsync(d_output, d_inputs[0], bytes, cudaMemcpyDeviceToDevice, stream));
        for (int i = 1; i < num_gpus; ++i) {
            vectorizedAddKernel<T><<<grid_size, threads_per_block, 0, stream>>>(d_output, d_inputs[i], elements);
            CUDA_CHECK(cudaGetLastError());
        }
    }
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    double avg_ms = (double)ms / bench_iters;

    // Cleanup
    for (int i = 0; i < num_gpus; ++i) {
        CUDA_CHECK(cudaFree(d_inputs[i]));
    }
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaStreamDestroy(stream));

    return avg_ms;
}

// ==============================================================================
// BENCHMARK EXECUTION COORDINATOR
// ==============================================================================
template <typename T>
void run_collective_benchmark(int num_gpus, size_t num_bytes, double baseline_ms, int bench_iters, int warmup_iters) {
    size_t elements = num_bytes / sizeof(T);
    ThreadBarrier barrier(num_gpus);
    
    std::vector<std::thread> threads;
    std::vector<double> nccl_times(num_gpus, 0.0);
    std::vector<bool> verification_results(num_gpus, false);
    
    ncclUniqueId ncclId;
    NCCL_CHECK(ncclGetUniqueId(&ncclId));
    
    for (int rank = 0; rank < num_gpus; ++rank) {
        threads.emplace_back([&, rank, elements, num_bytes, bench_iters, warmup_iters]() {
            CUDA_CHECK(cudaSetDevice(rank));
            
            cudaStream_t stream;
            CUDA_CHECK(cudaStreamCreate(&stream));
            
            ncclComm_t comm;
            NCCL_CHECK(ncclCommInitRank(&comm, num_gpus, ncclId, rank));
            
            T *d_send, *d_recv;
            CUDA_CHECK(cudaMalloc(&d_send, num_bytes));
            CUDA_CHECK(cudaMalloc(&d_recv, num_bytes));
            
            int num_sm = 0;
            CUDA_CHECK(cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, rank));
            int grid_size = num_sm * 4;
            
            initKernel<T><<<grid_size, 256, 0, stream>>>(d_send, elements, rank);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemsetAsync(d_recv, 0, num_bytes, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            
            barrier.wait();
            
            // Warmup iterations
            for (int w = 0; w < warmup_iters; ++w) {
                NCCL_CHECK(ncclAllReduce(d_send, d_recv, elements, TypeTraits<T>::nccl_type(), ncclSum, comm, stream));
            }
            CUDA_CHECK(cudaStreamSynchronize(stream));
            barrier.wait();
            
            cudaEvent_t start, stop;
            CUDA_CHECK(cudaEventCreate(&start));
            CUDA_CHECK(cudaEventCreate(&stop));
            
            barrier.wait();
            CUDA_CHECK(cudaEventRecord(start, stream));
            for (int b = 0; b < bench_iters; ++b) {
                NCCL_CHECK(ncclAllReduce(d_send, d_recv, elements, TypeTraits<T>::nccl_type(), ncclSum, comm, stream));
            }
            CUDA_CHECK(cudaEventRecord(stop, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            barrier.wait();
            
            float ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
            nccl_times[rank] = (double)ms / bench_iters;
            
            // Pinned page-locked memory allocations replace slow pageable memory
            T* host_recv = nullptr;
            CUDA_CHECK(cudaMallocHost((void**)&host_recv, num_bytes));
            CUDA_CHECK(cudaMemcpyAsync(host_recv, d_recv, num_bytes, cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            
            verification_results[rank] = verify_results<T>(host_recv, elements, num_gpus);
            CUDA_CHECK(cudaFreeHost(host_recv));
            
            CUDA_CHECK(cudaFree(d_send));
            CUDA_CHECK(cudaFree(d_recv));
            CUDA_CHECK(cudaEventDestroy(start));
            CUDA_CHECK(cudaEventDestroy(stop));
            NCCL_CHECK(ncclCommDestroy(comm));
            CUDA_CHECK(cudaStreamDestroy(stream));
        });
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    // Evaluate slowest-link latency (p99 boundary)
    double max_nccl_ms = 0.0;
    for (double t : nccl_times) {
        max_nccl_ms = std::max(max_nccl_ms, t);
    }
    
    bool all_passed = true;
    for (bool v : verification_results) {
        if (!v) all_passed = false;
    }
    
    // Algorithmic and Bus Bandwidth formulations matching standard NCCL definitions
    double algo_bw = (double)num_bytes / (max_nccl_ms * 1e6); // GB/s (using decimal GB metrics)
    double bus_bw = algo_bw * (2.0 * (num_gpus - 1) / num_gpus);
    
    // Weak scaling efficiency relative to single-GPU baseline execution: T_baseline / (N * T_allreduce)
    double efficiency = (baseline_ms) / (max_nccl_ms * num_gpus) * 100.0;
    
    std::string size_str;
    if (num_bytes >= 1024 * 1024 * 1024) {
        size_str = std::to_string(num_bytes / (1024 * 1024 * 1024)) + " GB";
    } else {
        size_str = std::to_string(num_bytes / (1024 * 1024)) + " MB";
    }
    
    printf("| %-8s | %-6s | %10zu | %15.3f | %10.3f | %12.3f | %12.3f | %10.2f%% | %-12s |\n",
           size_str.c_str(),
           TypeTraits<T>::name(),
           elements,
           baseline_ms,
           max_nccl_ms,
           algo_bw,
           bus_bw,
           efficiency,
           all_passed ? "SUCCESS" : "FAILED");
}

// ==============================================================================
// TOPOLOGY ASSESSMENT HOOKS
// ==============================================================================
void query_system_topology(int num_gpus) {
    printf("\n======================================================================\n");
    printf("PEER-TO-PEER (P2P) ACCESS MATRIX\n");
    printf("======================================================================\n");
    printf("     ");
    for (int j = 0; j < num_gpus; ++j) printf("GPU %-4d", j);
    printf("\n");
    for (int i = 0; i < num_gpus; ++i) {
        printf("GPU %d", i);
        for (int j = 0; j < num_gpus; ++j) {
            if (i == j) {
                printf("  SELF  ");
            } else {
                int can_access = 0;
                CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access, i, j));
                printf("   %s   ", can_access ? "OK " : "X  ");
            }
        }
        printf("\n");
    }
    printf("======================================================================\n");
    printf("Diagnostic Check: If 'X' is indicated, hardware ACS/IOMMU blocks direct\n");
    printf("                  BAR1 access paths over PCIe. Performance drops ~2X.\n");
    printf("======================================================================\n\n");
}

int main(int argc, char* argv[]) {
    // Inject runtime configurations to override driver defaults
    setenv("NCCL_DEBUG", "INFO", 1);
    setenv("NCCL_BUFFSIZE", "4194304", 1); 

    int system_gpus = 0;
    CUDA_CHECK(cudaGetDeviceCount(&system_gpus));

    if (system_gpus < 2) {
        fprintf(stderr, "[FATAL] Benchmark requires a minimum of 2 physical GPUs. Terminating execution.\n");
        return EXIT_FAILURE;
    }

    int run_gpus = system_gpus;
    if (run_gpus > 4) {
        run_gpus = 4; // Target execution context constraint
    }

    query_system_topology(run_gpus);

    // Warm up standard systems and enable peer access safely
    for (int i = 0; i < run_gpus; ++i) {
        CUDA_CHECK(cudaSetDevice(i));
        for (int j = 0; j < run_gpus; ++j) {
            if (i != j) {
                int can_access = 0;
                CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access, i, j));
                if (can_access) {
                    cudaError_t err = cudaDeviceEnablePeerAccess(j, 0);
                    if (err != cudaSuccess && err != cudaErrorPeerAccessAlreadyEnabled) {
                        CUDA_CHECK(err);
                    }
                }
            }
        }
    }

    printf("\nInitializing Multi-GPU NCCL AllReduce Evaluation Matrix (%d Devices)\n", run_gpus);
    printf("------------------------------------------------------------------------------------------------------------\n");
    printf("| Msg Size | Type   | Elements   | Baseline (1 GPU) | NCCL Time | Algo Bandwidth | Bus Bandwidth | Efficiency  | Verification |\n");
    printf("|          |        |            | (ms)             | (ms)      | (GB/s)         | (GB/s)        |             |              |\n");
    printf("------------------------------------------------------------------------------------------------------------\n");

    const int bench_iterations = 20;
    const int warmup_iterations = 10;
    
    // Ordered physical buffer allocations
    std::vector<size_t> sizes_bytes = {
        1 * 1024 * 1024,      // 1MB
        10 * 1024 * 1024,     // 10MB
        100 * 1024 * 1024,    // 100MB
        1024 * 1024 * 1024    // 1GB (Calculates peak staging limits)
    };

    for (size_t bytes : sizes_bytes) {
        // FP32 Passes
        double b_fp32 = run_single_gpu_baseline<float>(bytes / sizeof(float), run_gpus, bench_iterations, warmup_iterations);
        run_collective_benchmark<float>(run_gpus, bytes, b_fp32, bench_iterations, warmup_iterations);

        // FP16 Passes
        double b_fp16 = run_single_gpu_baseline<half>(bytes / sizeof(half), run_gpus, bench_iterations, warmup_iterations);
        run_collective_benchmark<half>(run_gpus, bytes, b_fp16, bench_iterations, warmup_iterations);

        // INT8 Passes
        double b_int8 = run_single_gpu_baseline<int8_t>(bytes / sizeof(int8_t), run_gpus, bench_iterations, warmup_iterations);
        run_collective_benchmark<int8_t>(run_gpus, bytes, b_int8, bench_iterations, warmup_iterations);
        
        printf("------------------------------------------------------------------------------------------------------------\n");
    }

    return EXIT_SUCCESS;
}
