import numpy as np

from analyticcenter import WeightedSystem

A = - 2*np.matrix([[1, -1], [1, 1]])
B = np.matrix([[1], [1]])
C = B.T
D = C @ B
Q = 0*np.identity(2)
sys = WeightedSystem(A, B, C, D, Q,  B, 2 * D)
