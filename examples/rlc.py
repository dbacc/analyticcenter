import control
import numpy as np
from analyticcenter.linearsystem import OptimalControlSystem

RR = 100.  # Resistor Value
L = 0.1e-1  # Inductance
C = 1.e-1 # Capacitor
# RR = 1.  # Resistor Value
# L = 1.  # Inductance
# C = 1. # Capacitor

p = 0  # Number of systems to connect

num = np.array([1 / (L * C)])
den = np.array([1, RR / L, 1 / (L * C)])
#
# num = np.array([1 / (RR * C), 0])
# den = np.array([1, 1 / (RR * C), 1 / (L * C)])


tf = control.tf(num, den)

ss = tf
for i in range(p):
    ss = control.series(ss, tf)

sys = control.tf2ss(ss)
A = sys.A
B = sys.B
C = np.asmatrix(sys.C)
D = np.asmatrix(sys.D) + 1


# D = np.matrix([1])
n = A.shape[0]
Q = np.zeros((n, n))
S = C.H
R = D + D.H
sys = OptimalControlSystem(A, B, C, D, Q, S, R)

# sys.check_passivity()
