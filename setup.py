# -*- encoding: utf-8 -*-
import os
import sys
from setuptools import setup, find_packages
from setuptools.extension import Extension
from setuptools.command.build_ext import build_ext


if sys.version_info < (3, 5):
    raise ValueError(
        'Unsupported Python version %d.%d.%d found. Mosaic requires Python '
        '3.5 or higher.' % (sys.version_info.major, sys.version_info.minor, sys.version_info.micro)
    )


class BuildExt(build_ext):
    """ build_ext command for use when numpy headers are needed.
    SEE tutorial: https://stackoverflow.com/questions/2379898
    SEE fix: https://stackoverflow.com/questions/19919905
    """

    def finalize_options(self):
        build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        # __builtins__.__NUMPY_SETUP__ = False
        import numpy
        self.include_dirs.append(numpy.get_include())


HERE = os.path.abspath(os.path.dirname(__file__))
setup_reqs = ['Cython', 'numpy']
with open(os.path.join(HERE, 'requirements.txt')) as fp:
    install_reqs = [r.rstrip() for r in fp.readlines()
                    if not r.startswith('#') and not r.startswith('git+')]

setup(
    name='mosaic_ml',
    author='Herilalaina Rakotoarison',
    author_email='herilalaina.rakotoarison@inria.fr',
    description='Mosaic for Machine Learning algorithm.',
    version="0.1-beta",
    cmdclass={'build_ext': BuildExt},
    packages=find_packages(exclude=['examples', 'test']),
    setup_requires=setup_reqs,
    install_requires=install_reqs,
    include_package_data=True,
    license='BSD',
    platforms=['Linux'],
    classifiers=[
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Information Technology",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    python_requires='>=3.5.*',
)
