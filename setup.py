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
    version='0.2.1',
    description='Computes the Analytic Center of a passive LTI continuous or discrete time system',
    author='Daniel Bankmann',
    author_email='bankmann@math.tu-berlin.de',
    url='https://gitlab.tubit.tu-berlin.de/PassivityRadius/analyticcenter',
    tests_require=['pytest'],
    data_files=[(_config_path, [join(_config_path, 'config.yaml')]),
                (_examples_path, glob.glob(_examples_path + '/*.npy')),
                (_examples_path, glob.glob(_examples_path + '/*.ipynb'))],
    license='BSD 3 License',
    packages=find_packages(exclude=["test", "test.*"]),
    platforms='all'
)
