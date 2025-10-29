
import paddle
from paddle.utils.cpp_extension import CUDAExtension, setup

def get_gencode_flags(compiled_all=False):
    if not compiled_all:
        prop = paddle.device.cuda.get_device_properties()
        cc = prop.major * 10 + prop.minor
        return ["-gencode", "arch=compute_{0},code=sm_{0}".format(cc)]
    else:
        return [
            "-gencode",
            "arch=compute_80,code=sm_80",
            "-gencode",
            "arch=compute_75,code=sm_75",
            "-gencode",
            "arch=compute_70,code=sm_70",
        ]

def get_sm_version():
    prop = paddle.device.cuda.get_device_properties()
    cc = prop.major * 10 + prop.minor
    return cc

cc_flag = get_gencode_flags(compiled_all=False)
cc = get_sm_version()

if cc >= 75:
    cc_flag.append("-DCUDA_BFLOAT16_AVAILABLE")

extra_compile_args = {
    "cxx": ["-O3"],
    "nvcc": [
        "-O3",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT162_OPERATORS__",
        "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
        "--ptxas-options=-v",
        "-lineinfo",
        "--threads",
        "4",
    ]
    + cc_flag,
}

setup(
    name='custom_setup_op_test',
    ext_modules=CUDAExtension(
        sources=['/root/paddlejob/gpfs/zhangweilong/CuLearn/custom_op/elementwise/relu_cuda.cc', '/root/paddlejob/gpfs/zhangweilong/CuLearn/custom_op/elementwise/relu_cuda.cu'],
        extra_compile_args=extra_compile_args
    )
)

