import torch
import torch.nn as nn
from loguru import logger
from torch.utils.cpp_extension import load

rms_norm_cuda_base = load(name='rms_norm_cuda_base', sources=['kernels/rms_norm_kernel_base.cu'], build_directory='kernels/build', verbose=True)
rms_norm_cuda_fast = load(name='rms_norm_cuda_fast', sources=['kernels/rms_norm_kernel_fast.cu'], build_directory='kernels/build', verbose=True)
rms_norm_cuda_fp16 = load(name='rms_norm_cuda_fp16', sources=['kernels/rms_norm_kernel_fp16.cu'], build_directory='kernels/build', verbose=True)

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        torch.manual_seed(42)
        self.weight = nn.Parameter(torch.randn(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        assert hidden_states.dim() == 3, 'hidden_states should be of shape (batch_size, query_length, model_dim)'
        assert hidden_states.size(-1) == self.weight.size(0), 'hidden_states should have the same last dimension as weight'
        assert hidden_states.shape[2] >= 4096 and hidden_states.shape[2] % 32 == 0, 'hidden_states should have a dimension divisible by 32 and at least 4096'
        assert hidden_states.dtype == torch.float32, 'hidden_states should be of dtype torch.float32'
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states

class RMSNormBase(RMSNorm):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__(hidden_size, eps=1e-6)
        self.rms_kernel = rms_norm_cuda_base.rms_norm_kernel

    def forward(self, hidden_states):
        assert hidden_states.dim() == 3, 'hidden_states should be of shape (batch_size, query_length, model_dim)'
        assert hidden_states.size(-1) == self.weight.size(0), 'hidden_states should have the same last dimension as weight'
        assert hidden_states.shape[2] >= 4096 and hidden_states.shape[2] % 32 == 0, 'hidden_states should have a dimension divisible by 32 and at least 4096'
        assert hidden_states.dtype == torch.float32, 'hidden_states should be of dtype torch.float32'
        batch_size, query_length, model_dim = hidden_states.shape
        output = torch.empty_like(hidden_states, dtype=hidden_states.dtype, device=hidden_states.device)
        self.rms_kernel(
            hidden_states, output, self.weight.detach(), batch_size * query_length, model_dim, self.variance_epsilon)
        return output

class RMSNormFast(RMSNormBase):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__(hidden_size, eps=1e-6)
        self.rms_kernel = rms_norm_cuda_fast.rms_norm_kernel

class RMSNormFP16(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        torch.manual_seed(42)
        self.weight = nn.Parameter(torch.randn(hidden_size).to(torch.float16))
        self.variance_epsilon = eps
        self.rms_kernel = rms_norm_cuda_fp16.rms_norm_kernel

    def forward(self, hidden_states):
        assert hidden_states.dim() == 3, 'hidden_states should be of shape (batch_size, query_length, model_dim)'
        assert hidden_states.size(-1) == self.weight.size(0), 'hidden_states should have the same last dimension as weight'
        assert hidden_states.shape[2] >= 4096 and hidden_states.shape[2] % 32 == 0, 'hidden_states should have a dimension divisible by 32 and at least 4096'
        assert hidden_states.dtype == torch.float16, 'hidden_states should be of dtype torch.float16'
        batch_size, query_length, model_dim = hidden_states.shape
        output = torch.empty_like(hidden_states, dtype=hidden_states.dtype, device=hidden_states.device)
        self.rms_kernel(
            hidden_states, output, self.weight.detach(), batch_size * query_length, model_dim, self.variance_epsilon)
        return output
