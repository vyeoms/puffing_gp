"""
Build:   pip install .
Dev:     pip install -e .

Builds two extensions:
  puffing_gp   -- CPU GP (CBLAS/LAPACK), always built
  puffing_gpcu -- CUDA GP (cuBLAS/cuSOLVER), built only when nvcc is available

Requires: libopenblas-dev, liblapacke-dev
  (apt install libopenblas-dev liblapacke-dev)
"""

import os, shutil, subprocess
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext as Pybind11BuildExt
import pybind11

# CPU extension (Linux only)

cpu_ext = Pybind11Extension(
    "puffing_gp",
    sources=["gp_kernel.c", "gp.c", "gp_bind.cpp"],
    language="c++",
    include_dirs=["/usr/include/x86_64-linux-gnu"],
    library_dirs=["/usr/lib/x86_64-linux-gnu/openblas-pthread"],
    libraries=["openblas", "lapacke", "mvec", "gomp"],
    extra_compile_args=["-O3", "-march=native", "-ffast-math", "-fopenmp"],
    extra_link_args=["-fopenmp"],
)

# CUDA extension (optional)

CUDA_ROOT  = os.environ.get("CUDA_ROOT", "/usr/local/cuda")
nvcc       = shutil.which("nvcc") or os.path.join(CUDA_ROOT, "bin", "nvcc")
HAS_CUDA   = os.path.isfile(nvcc)

if HAS_CUDA:
    cuda_ext = Pybind11Extension(
        "puffing_gpcu",
        sources=["gpcu_bind.cpp"],   # gp_cuda.cu and gp_cuda_kernel.cu compiled below
        include_dirs=[os.path.join(CUDA_ROOT, "include")],
        library_dirs=[os.path.join(CUDA_ROOT, "lib64")],
        libraries=["cublas", "cusolver", "cudart"],
        extra_compile_args=["-O3"],
    )
else:
    cuda_ext = None

# custom build_ext: pre-compile CUDA sources with nvcc

class MixedBuildExt(Pybind11BuildExt):
    def build_extension(self, ext):
        if ext.name == "puffing_gpcu":
            os.makedirs(self.build_temp, exist_ok=True)

            cuda_sources = ["gp_cuda_kernel.cu", "gp_cuda.cu"]
            cuda_objs    = []
            for src in cuda_sources:
                obj = os.path.join(self.build_temp, src.replace(".cu", ".o"))
                cmd = [
                    nvcc,
                    "-O3", "--use_fast_math",
                    "-Xcompiler", "-fPIC",
                    "-std=c++14",
                    f"-I{pybind11.get_include()}",
                    f"-I{CUDA_ROOT}/include",
                    "-I.", "-c", src, "-o", obj,
                ]
                print(" ".join(cmd))
                subprocess.check_call(cmd)
                cuda_objs.append(obj)

            ext.extra_objects = cuda_objs

        super().build_extension(ext)

# setup

ext_modules = [cpu_ext]
if cuda_ext:
    ext_modules.append(cuda_ext)

setup(
    name="puffing_gp",
    version="0.1.0",
    description="GP regression -- CBLAS/LAPACK + CUDA accelerated",
    py_modules=["gp_torch"],
    ext_modules=ext_modules,
    cmdclass={"build_ext": MixedBuildExt},
    python_requires=">=3.8",
    install_requires=["pybind11>=2.10", "numpy"],
)
