#!/usr/bin/env python

from setuptools import setup

setup(
    name='analyticcenter',
    version='0.1',
    description='Computes the Analytic Center of an LTI system',
    author='Daniel Bankmann',
    author_email='bankmann@math.tu-berlin.de',
    url='http://www.math.tu-berlin.de',
    setup_requires=['pytest-runner', 'colorlog==2.10.0', 'control==0.7.0', 'matplotlib==2.0.0', 'numpy==1.13.3',
                    'PyYAML==3.12', 'slycot==0.2.0'],
    tests_require=['pytest'],
    license='GNU',
    packages=['analyticcenter', 'test'],
    platforms='all'
)
