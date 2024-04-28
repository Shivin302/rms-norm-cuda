from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='rms_norm_cuda',
    ext_modules=[
        CUDAExtension('rms_norm_cuda_ext', [
            'rms_norm_kernel.cu'
        ], extra_compile_args={'cxx': ['-g', '-std=c++14'], 'nvcc': ['-O0']})
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
