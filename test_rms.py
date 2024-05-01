"""pytest module to test correctness of RMSNorm CUDA kernels"""
import torch
from rms_modules import RMSNorm, RMSNormBase, RMSNormFast, RMSNormFP16
import pytest

def test_correct():
    """
    Test correctness of RMSNorm CUDA kernels over a range of input sizes
    """
    test_dims = [
        (1, 1, 4096),
        (1, 1, 4096+32),
        (3, 7, 4096+256),
        (32, 500, 4096+256),
    ]
    for batch_size, query_length, model_dim in test_dims:
        arr = torch.randn(batch_size, query_length, model_dim).cuda()
        norm = RMSNorm(model_dim).cuda()
        arr_norm = norm.forward(arr)

        norm_base = RMSNormBase(model_dim).cuda()
        arr_norm_base = norm_base.forward(arr)
        diff = torch.mean(torch.abs(arr_norm - arr_norm_base))
        assert diff < 1e-5

        norm_fast = RMSNormFast(model_dim).cuda()
        arr_norm_fast = norm_fast.forward(arr)
        diff = torch.mean(torch.abs(arr_norm - arr_norm_fast))
        assert diff < 1e-5

        norm_fp16 = RMSNormFP16(model_dim).cuda()
        arr_norm_fp16 = norm_fp16.forward(arr.to(torch.float16))
        diff = torch.mean(torch.abs(arr_norm - arr_norm_fp16))
        assert diff < 1e-3
