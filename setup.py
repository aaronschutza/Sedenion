from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='sedenion_ops',
    ext_modules=[
        CUDAExtension(
            name='sedenion_ops', 
            sources=[
                'src/associator.cu',
                'src/bindings.cpp', 
            ],
            extra_compile_args={
                # Direct to MSVC for bindings.cpp
                'cxx': ['/O2', '/D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH'], 
                
                # Direct to NVCC for associator.cu
                'nvcc': [
                    '-allow-unsupported-compiler',
                    '-O3',
                    # Force NVCC to pass the macro directly to the underlying MSVC compiler
                    '-Xcompiler', '/D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH' 
                ] 
            }
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })