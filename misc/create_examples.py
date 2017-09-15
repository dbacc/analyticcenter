import numpy as np

from analyticcenter.linearsystem import OptimalControlSystem
from analyticcenter.algorithm import get_analytic_center_object
import control
from misc.misc import check_positivity


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


def generate_random_sys_and_save_2(m, n):
    while True:
        roots = -np.random.randint(1, 100, n)
        A = np.diag(roots)



        B = np.random.rand(n, m)
        C = np.random.rand(m, n)
        D = 10*np.random.rand(m, m)
        ss = control.ss( A, B, C, D)
        tf = control.ss2tf(ss)
        ss = control.tf2ss(tf)


        sys = OptimalControlSystem(ss.A, ss.B, ss.C, ss.D, np.zeros(n,n), ss.C.H, ss.D + ss.D.H)
        alg = AnalyticCenter(sys, 10 ** (-3))
        if sys._check_positivity(sys.H0):
            continue
        if sys._check_positivity(alg._get_H_matrix(alg._get_initial_X())):
            break

    sys.save()

def generate_random_sys_and_save_3(m, n):
    while True:
        print("new")



        ss = control.matlab.rss(n, m, m)


        ss.D = 100 * np.asmatrix(np.ones((m,m)))
        R = ss.D + ss.D.H
        ss.A = 10 * ss.A
        # ss.B = 1/10 * ss.B
        ss.C = 1/10 * ss.C
        sys = OptimalControlSystem(ss.A, ss.B, ss.C, ss.D, np.zeros((n,n)), ss.C.H, R)
        import ipdb
        ipdb.set_trace()
        X = control.care(ss.A, ss.B, np.zeros((n,n)), R, ss.C.H, np.identity(n))[0]

        alg = get_analytic_center_object(sys, 10 ** (-3))

        if check_positivity(-X):
            break

    sys.save()