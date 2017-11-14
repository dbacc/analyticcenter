import numpy as np
import analyticcenter
from analyticcenter import WeightedSystem
from analyticcenter import get_algorithm_object
from analyticcenter import NewtonDirectionMultipleDimensionsCT
from analyticcenter import SteepestAscentDirectionCT
sysmat = np.load('examples/example-n-6-m-3.npy')
sys = WeightedSystem(*sysmat)

if __name__=="__main__":
    alg = get_algorithm_object(sys, discrete_time=False, save_intermediate=True, abs_tol=9e-12)
    direction_method1 = NewtonDirectionMultipleDimensionsCT()
    direction_method1.maxiter = 40
    (ac, success) = direction_method1()
    direction_method2 = SteepestAscentDirectionCT()
    (ac, success) = direction_method2()