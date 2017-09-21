import control
import numpy as np
from analyticcenter.linearsystem import OptimalControlSystem
from scipy.signal import cheby2




num, den = cheby2(6,30, 100, analog=True)
tf = control.tf(num, den)

ss = tf


sys = control.tf2ss(ss)
A = sys.A
B = sys.B
C = np.asmatrix(sys.C)
D = np.asmatrix(sys.D)


# D = np.matrix([1])
n = A.shape[0]
Q = np.zeros((n, n))
S = C.H
R = D + D.H
sys = OptimalControlSystem(A, B, C, D, Q, S, R)

# sys.check_passivity()
