import torch
from loguru import logger
import sys
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from rms_modules import RMSNorm, RMSNormBase, RMSNormFast, RMSNormFP16

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

def debug_outputs():
    batch_size, query_length, model_dim = 2, 2, 4096
    arr = torch.randn(batch_size, query_length, model_dim).cuda()
    print(arr[:2,:2,:4])
    logger.info("Starting Pytorch RMSNorm")
    norm_torch = RMSNorm(model_dim).cuda()
    arr_norm = norm_torch.forward(arr)
    print(arr_norm[:2,:2,:4])
    logger.info("Finished Pytorch RMSNorm")

    logger.info("Starting Baseline RMSNorm")
    norm_baseline = RMSNormBase(model_dim).cuda()
    arr_norm_baseline = norm_baseline.forward(arr)
    print(arr_norm_baseline[:2,:2,:4])
    print(torch.mean(torch.abs(arr_norm - arr_norm_baseline)))
    logger.info("Finished Baseline RMSNorm")

    logger.info("Starting RMSNorm Fast")
    norm_fast = RMSNormFast(model_dim).cuda()
    arr_norm_fast = norm_fast.forward(arr)
    print(arr_norm_fast[:2,:2,:4])
    print(torch.mean(torch.abs(arr_norm - arr_norm_fast)))
    logger.info("Finished RMSNorm Fast")

    arr_16 = arr.to(torch.float16)
    logger.info("Starting RMSNorm FP16")
    norm_fp16 = RMSNormFP16(model_dim).cuda()
    arr_norm_fp16 = norm_fp16.forward(arr_16)
    print(arr_norm_fp16[:2,:2,:4])
    print(torch.mean(torch.abs(arr_norm - arr_norm_fp16)))
    logger.info("Finished RMSNorm FP16")

def run_rmsnorm():
    batch_size, nbatches = 32, 20
    query_length, model_dim = 1000, 4096
    arr = torch.randn(batch_size * nbatches, query_length, model_dim)
    dataset = TensorDataset(arr)
    norm_torch = RMSNorm(model_dim).cuda()
    norm_baseline = RMSNormBase(model_dim).cuda()
    norm_fast = RMSNormFast(model_dim).cuda()
    norm_fp16 = RMSNormFP16(model_dim).cuda()
    for norm in [norm_torch, norm_baseline, norm_fast]:
        logger.info("Starting forward pass")
        for b in range(0, batch_size*nbatches, batch_size):
            norm.forward(arr[b:b+batch_size].cuda())
        torch.cuda.synchronize()
        logger.info("Finished forward pass")
    
    arr_16 = arr.to(torch.float16)
    logger.info("Starting FP16 forward pass")
    for b in range(0, batch_size*nbatches, batch_size):
        norm_fp16.forward(arr_16[b:b+batch_size].cuda())
    torch.cuda.synchronize()
    logger.info("Finished FP16 forward pass")


if __name__ == "__main__":
    configure_logger()
    debug_outputs()
    run_rmsnorm()