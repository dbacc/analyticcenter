import numpy as np

from examples import OptimalControlSystem

sysmat = np.load('example-n-4-m-2.npy')
sys = OptimalControlSystem(*sysmat)