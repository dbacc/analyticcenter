import numpy as np

from analyticcenter import WeightedSystem

sysmat = np.load('test/test_examples/example-n-6-m-1.npy')
sys = WeightedSystem(*sysmat)
