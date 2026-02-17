from pathlib import Path

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

here = Path(__file__).resolve().parent
cuda_dir = here / 'cuda'

def src(*names):
    return [str(cuda_dir / name) for name in names]

ext_modules = [
    CUDAExtension('fixed_renderer.cuda.load_textures', src('load_textures_cuda.cpp', 'load_textures_cuda_kernel.cu')),
    CUDAExtension('fixed_renderer.cuda.rasterize', src('rasterize_cuda.cpp', 'rasterize_cuda_kernel.cu')),
    CUDAExtension('fixed_renderer.cuda.create_texture_image', src('create_texture_image_cuda.cpp', 'create_texture_image_cuda_kernel.cu')),
]

setup(
    name='fixed_renderer',
    version='0.0.1',
    packages=['fixed_renderer', 'fixed_renderer.cuda'],
    package_dir={'fixed_renderer': '.'},
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension},
)
