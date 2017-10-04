import numpy as np

from analyticcenter.linearsystem import WeightedSystem
sysmat = np.load('examples/example-n-6-m-3.npy')
sys = WeightedSystem(*sysmat)