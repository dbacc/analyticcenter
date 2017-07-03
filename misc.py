from analyticcenter.linearsystem import OptimalControlSystem
import numpy as np
from scipy import linalg
from analyticcenter.algorithm import AnalyticCenter


def rsolve(*args, **kwargs):
    return np.asmatrix(linalg.solve(np.asmatrix(args[0]).H, np.asmatrix(args[1]).H, kwargs)).H


def schur_complement(X, n, mode='upper'):
    s = X.shape[0]
    if mode.lower() == 'upper':
        complement = X[0:n, 0:n] - X[0:n, n:s] @ linalg.inv(X[n:s, n:s]) @ X[n:s, 0:n]
    else:
        raise NotImplementedError("Schur Complement currently only works for upper part")
    return complement


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
