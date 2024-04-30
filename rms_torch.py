import torch
import torch.nn as nn
from loguru import logger
import sys
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
from torch.utils.cpp_extension import load

rms_norm_cuda_base = load(name='rms_norm_cuda_base', sources=['kernels/rms_norm_kernel_base.cu'], build_directory='kernels/build', verbose=True)
rms_norm_cuda_fast = load(name='rms_norm_cuda_fast', sources=['kernels/rms_norm_kernel_fast.cu'], build_directory='kernels/build', verbose=True)

def configure_logger():
    default_format = " | ".join([
        "<green>{time:YYYY-MM-DD HH:mm:ss.SS}</green>",
        "<cyan>{module: >14.14}</cyan>:<cyan>{line: <4}</cyan> <cyan>{function: <16.16}</cyan>",
        "<level>{level.icon} {message}</level>",
    ])
    logger.configure(handlers=[dict(sink=sys.stderr, format=default_format, level='INFO')])

    logger.level("SUCCESS", icon="✅")
    logger.level("WARNING", icon="🟡")
    logger.level("INFO", icon="ℹ︎")

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        torch.manual_seed(42)
        self.weight = nn.Parameter(torch.randn(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class RMSNormCUDA(RMSNorm):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__(hidden_size, eps=1e-6)
        self.rms_kernel = rms_norm_cuda_base.rms_norm_kernel

    def forward(self, hidden_states):
        batch_size, query_length, model_dim = hidden_states.shape
        hidden_states = hidden_states.to(torch.float32)
        output = torch.empty_like(hidden_states, dtype=hidden_states.dtype, device=hidden_states.device)
        self.rms_kernel(
            hidden_states, output, self.weight.detach(), batch_size * query_length, model_dim, self.variance_epsilon)
        return output.to(hidden_states.dtype)

class RMSNormCUDAFast(RMSNormCUDA):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__(hidden_size, eps=1e-6)
        self.rms_kernel = rms_norm_cuda_fast.rms_norm_kernel

def main():
    batch_size, query_length, model_dim = 32, 1000, 4096
    arr = torch.randn(batch_size, query_length, model_dim).cuda()
    # arr = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]).cuda()
    print(arr[:2,:2,:2])
    logger.info("Starting Pytorch RMSNorm")
    norm = RMSNorm(model_dim).cuda()
    arr_norm = norm.forward(arr)
    print(arr_norm[:2,:2,:2])
    logger.info("Finished Pytorch RMSNorm")

    logger.info("Starting CUDA RMSNorm")
    norm = RMSNormCUDA(model_dim).cuda()
    arr_norm_cuda = norm.forward(arr)
    print(arr_norm_cuda[:2,:2,:2])
    print(torch.mean(torch.abs(arr_norm - arr_norm_cuda)))
    logger.info("Finished CUDA RMSNorm")

    logger.info("Starting CUDA RMSNorm Fast")
    norm = RMSNormCUDAFast(model_dim).cuda()
    arr_norm_cuda = norm.forward(arr)
    print(arr_norm_cuda[:2,:2,:2])
    print(torch.mean(torch.abs(arr_norm - arr_norm_cuda)))
    logger.info("Finished CUDA RMSNorm Fast")

if __name__ == "__main__":
    configure_logger()
    main()