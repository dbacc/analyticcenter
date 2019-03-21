##
## Copyright (c) 2017
##
## @author: Daniel Bankmann
## @company: Technische Universität Berlin
##
## This file is part of the python package analyticcenter
## (see https://gitlab.tu-berlin.de/PassivityRadius/analyticcenter/)
##
## License: 3-clause BSD, see https://opensource.org/licenses/BSD-3-Clause
##
import numpy as np
from scipy import linalg
import logging
import control
import scipy.io



class LTI(object):
    """Describes an LTI system with system matrices of the appropriate sizes"""

    def __init__(self, A, B, C, D):
        self.A = np.asmatrix(A)
        self.B = np.asmatrix(B)
        self.C = np.asmatrix(C)
        self.D = np.asmatrix(D)
        self.n = A.shape[0]
        self.m = B.shape[1]
        self.p = C.shape[0]
        self.logger = logging.getLogger(__name__)


class WeightedSystem(LTI):
    """Describes LTI system with weight Matrix, where Q = Q.H, R = R.H"""

    def __init__(self, A, B, C, D, Q, S, R):
        super().__init__(A, B, C, D)
        self.Q = np.asmatrix(Q)
        self.S = np.asmatrix(S)
        self.R = np.asmatrix(R)
        self.Rsqrtm = np.asmatrix(linalg.sqrtm(self.R))
        self.__initH0()


    def save(self):
        np.save('example-n-{}-m-{}'.format(self.n, self.m), [self.A, self.B, self.C, self.D, self.Q, self.S, self.R])

    def save_mat(self):
        scipy.io.savemat('example-n-{}-m-{}'.format(self.n, self.m),
                         {"A": self.A, "B": self.B, "C": self.C, "D": self.D, "Q": self.Q, "S": self.S, "R": self.R})

    def __initH0(self):
        self.H0 = np.bmat([[self.Q, self.S], [self.S.transpose(), self.R]])

    def bilinear_discretization(self):
        identity = np.identity(self.n)
        identitym = np.identity(self.m)
        idamin = np.linalg.inv(self.A - identity ) #TODO: Make more robust
        Ad = np.linalg.solve(self.A - identity, self.A + identity)
        Bd = np.linalg.solve(self.A - identity, self.B) * np.sqrt(2)
        Tc = np.block([[ -np.sqrt(2) * idamin , - Bd /np.sqrt(2)],
                       [ np.zeros((self.m, self.n)), identitym]])
        Wc = np.block([[ self.Q, self.S],
                       [self.S.H, self.R]])
        Wd = Tc.H @ Wc @ Tc
        Qd = Wd[:self.n, :self.n]
        Qd = 0.5*(Qd + Qd.H)
        Sd = Wd[:self.n, self.n:]
        Rd = Wd[self.n:, self.n:]
        Rd = 0.5*(Rd + Rd.H)
        Cd = Sd.H
        Dd = Rd/2
        return WeightedSystem(Ad,Bd,Cd,Dd,Qd,Sd,Rd)
