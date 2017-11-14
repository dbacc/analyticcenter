
from analyticcenter import get_algorithm_object
from analyticcenter import NewtonDirectionMultipleDimensionsCT
from  analyticcenter import SteepestAscentDirectionCT

from analyticcenter.examples.example2 import sys
import numpy as np

if __name__ == "__main__":

    alg = get_algorithm_object(sys, discrete_time=False, save_intermediate=True, abs_tol=9e-12)
    direction_method1 = NewtonDirectionMultipleDimensionsCT()
    direction_method1.maxiter = 40
    (ac, success) = direction_method1()
    X0 =np.load("results/X_last.npy")
    direction_method2 = SteepestAscentDirectionCT()
    direction_method2.maxiter = 100000000
    (ac, success) = direction_method2(X0 = X0)
