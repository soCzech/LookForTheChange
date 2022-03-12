from setuptools import setup
from torch.utils import cpp_extension

library_dirs = cpp_extension.library_paths(cuda=True)
include_dirs = cpp_extension.include_paths(cuda=True)

print("library_dirs:", library_dirs)
print("include_dirs:", include_dirs)

setup(
    name="lookforthechange",
    version="1.0",
    install_requires=[
        "numpy",
        "torch"
    ],
    ext_modules=[
        cpp_extension.CUDAExtension(
            name='_lookforthechange_ops',
            sources=[
                'src/common.cpp',
                'src/optimal_state_change.cu'
            ],
            library_dirs=library_dirs,
            include_dirs=include_dirs
        )
    ],
    packages=['lookforthechange'],
    package_dir={'lookforthechange': './python'},
    cmdclass={'build_ext': cpp_extension.BuildExtension}
)
