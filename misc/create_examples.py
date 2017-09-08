import numpy as np

from analyticcenter.linearsystem import OptimalControlSystem
from analyticcenter.algorithm import AnalyticCenter

def generate_random_sys_and_save(m, n):
    while True:
        A = np.random.rand(n, n)
        B = np.random.rand(n, m)
        C = np.random.rand(m, n)
        D = np.random.rand(m, m)
        Q = np.random.rand(n, n)
        Q = Q @ Q.T
        S = 0.01 * np.random.rand(n, m)
        R = np.random.rand(m, m)
        R = R @ R.T
        sys = OptimalControlSystem(A, B, C, D, Q, S, R)
        alg = AnalyticCenter(sys, 10 ** (-3))
        if sys._check_positivity(sys.H0):
            continue
        if sys._check_positivity(alg._get_H_matrix(alg._get_initial_X())):
            break

    sys.save()
