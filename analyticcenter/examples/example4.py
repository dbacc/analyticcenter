import numpy as np
import analyticcenter
from analyticcenter import WeightedSystem
from analyticcenter import get_algorithm_object
from analyticcenter import NewtonDirectionMultipleDimensionsCT
from analyticcenter import SteepestAscentDirectionCT
sysmat = np.load('examples/example-n-30-m-10.npy')
sys = WeightedSystem(*sysmat)
