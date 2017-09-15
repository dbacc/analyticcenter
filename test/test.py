import inspect
import os

import yaml

from analyticcenter.algorithm import get_analytic_center_object
from analyticcenter.direction import NewtonDirectionMultipleDimensionsCT

from logger import prepare_logger

from examples.example1 import sys
from examples.rlc import sys as sys2



def test_1():
    alg = get_analytic_center_object(sys, 10 ** (-8), discrete_time=False)
    direction_method = NewtonDirectionMultipleDimensionsCT()
    (X, success) = direction_method()
    assert success


def test_1():
    alg = get_analytic_center_object(sys2, 10 ** (-8), discrete_time=False)
    direction_method = NewtonDirectionMultipleDimensionsCT()
    (X, success) = direction_method()
    assert success