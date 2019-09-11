from distutils.core import setup
from Cython.Build import cythonize

setup(name='cython_metrics', ext_modules=cythonize("cython_metrics.pyx"), zip_safe = False)