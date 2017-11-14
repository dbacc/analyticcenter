import pytest
import analyticcenter
from analyticcenter import get_algorithm_object
from analyticcenter import NewtonDirectionMultipleDimensionsCT, NewtonDirectionMultipleDimensionsDT
from analyticcenter import AnalyticCenterUnstable, AnalyticCenterNotPassive, \
    AnalyticCenterRiccatiSolutionFailed, AnalyticCenterUncontrollable

from analyticcenter.examples.rlc import sys as sysrlc
from test.test_examples.example1 import sys
from test.test_examples.example4 import sys as sysuncontrollable
from analyticcenter.examples.cheby_filter import sys as syscheby


def test_unstable_ct():
    with pytest.raises(AnalyticCenterUnstable):
        alg = get_algorithm_object(sys, discrete_time=False)
        direction_method = NewtonDirectionMultipleDimensionsCT()
        (X, success) = direction_method()


def test_unstable_dt():
    with pytest.raises(AnalyticCenterUnstable):
        alg = get_algorithm_object(sys, discrete_time=True)
        direction_method = NewtonDirectionMultipleDimensionsDT()
        (X, success) = direction_method()


def test_uncontrollable_ct():
    with pytest.raises(AnalyticCenterUncontrollable):
        alg = get_algorithm_object(sysuncontrollable, discrete_time=False)
        direction_method = NewtonDirectionMultipleDimensionsCT()
        (X, success) = direction_method()


def test_2():
    alg = get_algorithm_object(sysrlc, discrete_time=False, abs_tol=9e-1)
    direction_method = NewtonDirectionMultipleDimensionsCT()
    (X, success) = direction_method()
    assert success


def test_cheby_no_riccati_solution():
    with pytest.raises(AnalyticCenterRiccatiSolutionFailed):
        alg = get_algorithm_object(syscheby, discrete_time=False)
        direction_method = NewtonDirectionMultipleDimensionsCT()
        (X, success) = direction_method()
