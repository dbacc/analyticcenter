#!/usr/bin/env python

from setuptools import setup

setup(
    name='analyticcenter',
    version='0.1',
    description='Computes the Analytic Center of an LTI system',
    author='Daniel Bankmann',
    author_email='bankmann@math.tu-berlin.de',
    url='http://www.math.tu-berlin.de',
    setup_requires=['pytest-runner', 'slycot', 'numpy', 'scipy'],
    tests_require=['pytest'],
    license='GNU',
    platforms='all'
)


