from codecs import open
from os import path

from setuptools import setup

print("Installing metric_fa\n")

from Cython.Distutils import Extension
from Cython.Build import build_ext, cythonize

print('Yes\n')
ext_modules = cythonize([Extension('metric_fa_util', ['metric_fa_util.py'], cython_directives={'language_level' : 3})])
cmdclass = {'build_ext': build_ext}
opts = {"ext_modules": ext_modules, "cmdclass": cmdclass}

print(">>>> Starting to install!\n")

setup(
    name='metric_fa',
    install_requires=['numpy', 'scipy', 'tqdm'],
    extras_require={
        'networkx': ['networkx'],
        'igraph': ['python-igraph']
    },
    include_package_data=True,
    **opts
)