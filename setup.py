#!/usr/bin/env python3

from distutils.core import setup

install_requires = ["requests", "biopython", "numpy", "scipy", "scikit-learn"]

try:
    import tensorflow
except ImportError:
    install_requires.append("tensorflow")

setup(name='contactlib',
      version='2.1.1',
      description='Tool for searching proteins through structural fingerprints.',
      url='https://github.com/Chenyao2333/contactlib',
      packages=['contactlib', 'contactlib.search'],
      python_requires = ">=2.7",
      install_requires = install_requires
)
