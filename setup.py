#!/usr/bin/env python
##
## Copyright (c) 2017
##
## @author: Daniel Bankmann
## @company: Technische Universität Berlin
##
## This file is part of the python package analyticcenter
## (see https://gitlab.tu-berlin.de/PassivityRadius/analyticcenter/)
##
## License: 3-clause BSD, see https://opensource.org/licenses/BSD-3-Clause
##

from setuptools import setup, find_packages
import glob
from os.path import join

_base_name = 'analyticcenter'
_config_name = 'config'
_examples_name = 'examples'
_config_path = join(_base_name, _config_name)
_examples_path = join(_base_name, _examples_name)

setup(
    name=_base_name,
    version='0.2',
    description='Computes the Analytic Center of an LTI system',
    author='Daniel Bankmann',
    author_email='bankmann@math.tu-berlin.de',
    url='http://www.math.tu-berlin.de',
    setup_requires=['pytest-runner', 'colorlog>=2.10.0', 'control>=0.7.0', 'matplotlib>=2.0.0', 'numpy>=1.13.3',
                    'PyYAML>=3.12', 'slycot>=0.3.3', 'scipy'],
    tests_require=['pytest'],
    data_files=[(_config_path, [join(_config_path, 'config.yaml')]),
                (_examples_path, glob.glob(_examples_path + '/*.npy'))],
    license='GNU',
    packages=find_packages(exclude=["test", "test.*"]),
    platforms='all'
)
