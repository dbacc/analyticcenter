import numpy as np

from analyticcenter.linearsystem import WeightedSystem

A = np.matrix([[-1]])
B = np.matrix([[1]])
C = B.T
D = C @ B
Q = np.zeros((1,1))
sys = WeightedSystem(A, B, C, D, Q, B, D + D.T)
