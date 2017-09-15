import numpy as np

from analyticcenter.linearsystem import OptimalControlSystem

sysmat = np.load('examples/example-n-6-m-1.npy')
sys = OptimalControlSystem(*sysmat)