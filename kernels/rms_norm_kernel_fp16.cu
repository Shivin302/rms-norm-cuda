#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <vector>
#include <pybind11/pybind11.h>
#include <math.h>

namespace py = pybind11;

/**
 * @brief Computes the root mean square (RMS) normalization of the input tensor using CUDA.
 *
 * @details
 * The function operates on a 3D Tensor `input` of shape (batch_size, query_length, model_dim) and 1D Tensor 'weight' of shape (model_dim).
 * The RMS normalization is calculated as follows:
 * - Each element is divided by the RMS norm calculated along the third dimension.
 * - The result is then multiplied by the corresponding `weight` element along the third dimension.
 * - We use half precision (FP16) for most operations except FP32 for the inverse square root.
 *
 * @param input The input tensor to be normalized. It is a 3D tensor of shape (batch_size, query_length, model_dim).
 * @param output The output tensor to store the normalized values. It should have the same shape as the input tensor.
 * @param weight The weight tensor to multiply the normalized values with. It is a 1D tensor of shape (model_dim,).
 * @param num_queries The number of queries in the input tensor. It corresponds to the second dimension of the input tensor.
 * @param model_dim The dimension of the model. It corresponds to the third dimension of the input tensor.
 * @param eps A small value added to the denominator to avoid division by zero.
 */
__global__ void rms_norm_kernel_fp16(
                const half *__restrict__ input,
                half *__restrict__ output,
                const half *__restrict__ weight,
                const int num_queries,
                const int model_dim,
                const float eps) {

    // Each block processes one query at a time, upto num_queries / gridDim.x
    for (int query_idx = blockIdx.x; query_idx < num_queries; query_idx += gridDim.x) {
        extern __shared__ half shared[]; 
        // Each thread pulls corresponding squared elements to shared memory and sums them
        shared[threadIdx.x] = __float2half(0.0f);
        for (int j = threadIdx.x; j < model_dim; j += blockDim.x) {
            shared[threadIdx.x] = __hadd(shared[threadIdx.x], __hmul(input[query_idx * model_dim + j], input[query_idx * model_dim + j]));
        }
        __syncthreads();

        // Parallel reduction in half
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
                shared[threadIdx.x] = __hadd(shared[threadIdx.x], shared[threadIdx.x + s]);
            }
            __syncthreads();
        }
        float squared_sum = __half2float(shared[0]);

        // Compute the RMS normalization factor in float for stability, then cast it to half
        half inv_rms_norm = __float2half(rsqrtf((squared_sum / model_dim) + eps));

        // Normalize each element of input for the thread's query, and multiply by the weight
        for (int j = threadIdx.x; j < model_dim; j += blockDim.x) {
            output[query_idx * model_dim + j] = __hmul(__hmul(input[query_idx * model_dim + j], inv_rms_norm), weight[j]);
        }
    }
}


void rms_norm_wrapper_fp16(torch::Tensor input, torch::Tensor output, torch::Tensor weight, int num_queries, int model_dim, float eps)
{
    int devId;
    cudaGetDevice(&devId);

    int numSMs;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, devId);

    const int blocksPerSM = 32;
    const int totalBlocks = numSMs * blocksPerSM;
    const int threads = 256;
    const int shared_mem_size = threads * sizeof(float) / 2;

    rms_norm_kernel_fp16<<<totalBlocks, threads, shared_mem_size>>>(
        reinterpret_cast<half*>(input.data_ptr<at::Half>()),
        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
        reinterpret_cast<half*>(weight.data_ptr<at::Half>()),
        num_queries,
        model_dim,
        eps);
}

PYBIND11_MODULE(rms_norm_cuda_fp16, m)
{
    m.def("rms_norm_kernel", &rms_norm_wrapper_fp16, "RMS normalization kernel using FP16 (CUDA)");
}
