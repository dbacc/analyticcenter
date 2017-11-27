import numpy as np
from os.path import join, dirname
import analyticcenter
from analyticcenter import WeightedSystem, get_algorithm_object
from analyticcenter import NewtonDirectionMultipleDimensionsCT
from analyticcenter import SteepestAscentDirectionCT
sysmat = np.load(join(dirname(__file__), 'example-n-30-m-10.npy'))
sys = WeightedSystem(*sysmat)
