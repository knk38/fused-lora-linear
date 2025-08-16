from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="fused_lora_linear_cuda",
    ext_modules=[
        CUDAExtension(
            name="fused_lora_linear._cuda",
            sources=[
                "src/fused_lora_linear/kernels/cuda/bindings.cpp",
                "src/fused_lora_linear/kernels/cuda/lora_linear_kernel.cu",
            ],
            extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
