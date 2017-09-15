import numpy as np

from analyticcenter.linearsystem import OptimalControlSystem

A = np.matrix([[1, -1], [1, 1]])
B = np.matrix([[1], [1]])
C = B.T
D = C @ B
Q = np.identity(2)
sys = OptimalControlSystem(A, B, C, D, Q, 0 * B, 2 * D)
