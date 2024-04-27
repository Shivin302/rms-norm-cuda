import torch
import torch.nn as nn
from loguru import logger
import sys
import os

def configure_logger():
    default_format = " | ".join([
        "<green>{time:YYYY-MM-DD HH:mm:ss.SS}</green>",
        "<cyan>{module: >14.14}</cyan>:<cyan>{line: <4}</cyan> <cyan>{function: <16.16}</cyan>",
        "<level>{level.icon} {message}</level>",
    ])
    logger.configure(handlers=[dict(sink=sys.stderr, format=default_format, level='INFO')])

    # logger = logger.opt(colors=True)
    # logger.opt = partial(logger.opt, colors=True)

    logger.level("SUCCESS", icon="✅")
    logger.level("WARNING", icon="🟡")
    logger.level("INFO", icon="ℹ︎")


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

def main():
    logger.info("Testing RMSNorm")
    batch_size = 32
    query_length = 100
    model_dim = 4096
    norm = RMSNorm(model_dim)
    arr = torch.randn(batch_size, query_length, model_dim)
    norm.forward(arr)
    

if __name__ == "__main__":
    configure_logger()
    main()