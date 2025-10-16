from paddle.utils.cpp_extension import CUDAExtension, setup

extra_compile_args = {
    'cxx': ['-O2', '-std=c++17'],  # C++编译选项
    'nvcc': ['-O2', '-std=c++17']  # NVCC编译选项
}
setup(
    name='custom_setup_ops',
    ext_modules=CUDAExtension(
        sources=['relu_cuda.cc', 'relu_cuda.cu'],
        extra_compile_args=extra_compile_args
    )
)

