from analyticcenter.linearsystem import OptimalControlSystem
import numpy as np
from scipy import linalg
import ipdb


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


def test_spectral_radius(A, B):
    flag = False
    n = A.shape[0]
    identity = np.identity(n)
    Z = B @ B.H
    AAH = A @ A.H
    # T = np.asmatrix(linalg.eigh(AAH + Z)[1])
    # ipdb.set_trace()
    # AAH = T.H @ AAH @ T
    # Z = T.H @ Z @ T
    # A = T.H @ A @ T

    lhs = np.kron(identity, AAH + Z) + np.kron(AAH + Z, identity)
    rhs = np.kron(np.conj(A), A.H) + np.kron(A.T, A)
    itermatrix = linalg.solve(lhs, rhs)
    print("Spectral radius of lhs: {}".format(spectral_radius(lhs)))
    print("Spectral radius of rhs: {}".format(spectral_radius(rhs)))
    print("Spectral radius of itermatrix: {}".format(spectral_radius(itermatrix)))
    # ipdb.set_trace()
    if flag:

        ipdb.set_trace()
    alpha = 10
    lhs = lhs + alpha * np.identity(n**2)
    rhs = rhs + alpha * np.identity(n**2)
    itermatrix = linalg.solve(lhs, rhs)
    if flag:
        print("with shift alpha = {}".format(alpha))
        print("Spectral radius of lhs: {}".format(spectral_radius(lhs)))
        print("Spectral radius of rhs: {}".format(spectral_radius(rhs)))
        print("Spectral radius of itermatrix: {}".format(spectral_radius(itermatrix)))
        ipdb.set_trace()
    return np.less_equal(spectral_radius(itermatrix), 1. + 10**(-4))





def spectral_radius(A):
    eigs = linalg.eig(A)[0]
    return np.max(np.abs(eigs))



def random_test():
    for i in range(1000):
        dim = 10
        totalsize = dim
        jordanblocks = np.random.randint(1,dim)
        blocksizes = np.array(list(random_ints_with_sum(dim)))




        randA = np.asmatrix(np.asmatrix(linalg.block_diag(*list(create_jordan_blocks(blocksizes)))))
        trans = np.asmatrix(np.random.random((dim,dim)))
        if np.linalg.cond(trans) >1000:
            continue
        print(randA)
        # randA = linalg.solve(trans, randA) @ trans
        print(randA)
        if np.isclose(linalg.det(randA),0):
            continue
        if test_spectral_radius(randA, np.asmatrix(np.zeros((dim,dim)))):
            print("all ok")
        else:
            print("stop")
            ipdb.set_trace()

    ipdb.set_trace()


def random_ints_with_sum(n):
    """
    Generate non-negative random integers summing to `n`.
    """
    while n >=1:
        r = np.random.randint(1, n+1)
        yield r
        n -= r

def create_jordan_blocks(blocks):
    for blocksize in blocks:
        J = np.diag(np.ones(blocksize-1),1)  + np.diag( np.ones(blocksize)*(np.random.randint(-10,10) + 1j*np.random.randint(-10,10) ))

        yield J
