import numpy as np


sysmat = np.load('example-n-4-m-2.npy')
sys = OptimalControlSystem(*sysmat)