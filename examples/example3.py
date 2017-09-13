import numpy as np

from analyticcenter.linearsystem import OptimalControlSystem

A = np.matrix([[1, -1], [1, 1]])
B = np.matrix([[1], [1]])
C = B.T
D = C @ B
Q = np.zeros((2,2))
sys = OptimalControlSystem(A, B, C, D, Q, B, D+D.T)
