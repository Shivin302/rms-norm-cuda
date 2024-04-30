
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
 *
 * @param input The input tensor to be normalized. It is a 3D tensor of shape (batch_size, query_length, model_dim).
 * @param output The output tensor to store the normalized values. It should have the same shape as the input tensor.
 * @param weight The weight tensor to multiply the normalized values with. It is a 1D tensor of shape (model_dim,).
 * @param num_queries The number of queries in the input tensor. It corresponds to the second dimension of the input tensor.
 * @param model_dim The dimension of the model. It corresponds to the third dimension of the input tensor.
 * @param eps A small value added to the denominator to avoid division by zero.
 */
__global__ void rms_norm_kernel(
                const float *__restrict__ input,
                float *__restrict__ output,
                const float *__restrict__ weight,
                const int num_queries,
                const int model_dim,
                const float eps) {
    // Each block processes one query at a time, upto num_queries / gridDim.x
    for (int query_idx = blockIdx.x; query_idx < num_queries; query_idx += gridDim.x) {
        // Each thread pulls query to shared memory
        extern __shared__ float shared[];
        if (query_idx < num_queries) {
            // Each thread pulls corresponding squared elements to shared memory and sums them
            shared[threadIdx.x] = 0;
            for (int j = threadIdx.x; j < model_dim; j += blockDim.x) {
                shared[threadIdx.x] += input[query_idx * model_dim + j] * input[query_idx * model_dim + j];
            }
            __syncthreads();
            // parellel reduction to sum shared memory
            for (int s = blockDim.x / 2; s > 0; s >>= 1) {
                if (threadIdx.x < s) {
                    shared[threadIdx.x] += shared[threadIdx.x + s];
                }
                __syncthreads();
            }
            // Compute the rms norm for the query
            float inv_rms_norm = rsqrtf((shared[0] / model_dim) + eps);

            // Normalize each element of input for the thread's query, and multiply by the weight
            for (int j = threadIdx.x; j < model_dim; j += blockDim.x) {
                output[query_idx*model_dim + j] = input[query_idx*model_dim + j] * inv_rms_norm * weight[j];
            }
        }
    __syncthreads();
    }
}


void rms_norm_wrapper(torch::Tensor input, torch::Tensor output, torch::Tensor weight, int num_queries, int model_dim, float eps)
{
    int devId;
    cudaGetDevice(&devId);

    int numSMs;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, devId);

    const int blocks = numSMs * 32;
    const int threads = 1024;
    const int shared_mem_size = threads * sizeof(float);

    rms_norm_kernel<<<blocks, threads, shared_mem_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        weight.data_ptr<float>(),
        num_queries,
        model_dim,
        eps);
}

PYBIND11_MODULE(rms_norm_cuda_fast, m)
{
    m.def("rms_norm_kernel", &rms_norm_wrapper, "RMS normalization kernel (CUDA)");
}
