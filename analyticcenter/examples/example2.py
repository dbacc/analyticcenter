import numpy as np
import analyticcenter
from os.path import join, dirname
from analyticcenter import WeightedSystem, get_algorithm_object
from analyticcenter import NewtonDirectionMultipleDimensionsCT
from analyticcenter import SteepestAscentDirectionCT
sysmat = np.load(join(dirname(__file__),'example-n-6-m-3.npy'))
sys = WeightedSystem(*sysmat)

if __name__=="__main__":
    alg = get_algorithm_object(sys, 'newton', discrete_time=False, save_intermediate=True)
    alg()