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
import logging

import control
import numpy as np
from scipy import linalg

from .exceptions import AnalyticCenterNotPassive, AnalyticCenterUncontrollable, AnalyticCenterUnstable, \
    AnalyticCenterRiccatiSolutionFailed
from ..misc.control import place
from ..misc.misc import symmetric_product_pos_def, check_positivity

logger = logging.getLogger(__name__)


class RiccatiOperator(object):
    """Base RiccatiOperator class. Provides Templates for the functions needed later on"""
    debug = True
    logger = logging.getLogger(__name__)

    def __init__(self, system):
        self.system = system
        self.X0 = None
        self.H0 = None
        self.H = None
        self.__init_H0()
        self.check_stability()

    def __init_H0(self):
        """Initializes the inhomogeinity of the inequality"""
        self.H0 = np.bmat([[self.system.Q, self.system.S], [self.system.S.H, self.system.R]])

    def _get_Delta_H(self, Delta_X):
        """

        Parameters
        ----------
        Delta_X : Change in X. Assumed to be Hermitian.


        Returns
        -------
        Change in H(X).

        """
        return self._get_H_matrix(Delta_X) - self.H0

    def gradient_sweep(self, X):
        """
        Generates gradients in every basis direction and computes the maximal ascent among those. Can be used for
        debugging purposes

        Parameters
        ----------
        X : Current solution


        Returns
        -------
        index of direction (i,j) in which ascent is maximal
        """
        H = self._get_H_matrix(X)
        Hinv = np.asmatrix(linalg.inv(H))
        dimension = self.system.n
        grad = np.zeros((dimension, dimension))
        for i in np.arange(dimension):
            for j in np.arange(i + 1):
                E = np.zeros((dimension, dimension))
                if i == j:
                    E[i, i] = 1
                else:
                    E[i, j] = 1 / np.sqrt(2)
                    E[j, i] = 1 / np.sqrt(2)
                Delta_H = self._get_Delta_H(E)
                grad[i, j] = np.trace(- Hinv.H @ Delta_H)
                det_new = linalg.det(H + Delta_H)
                self.logger.debug("Gradient: {} Direction: {}, {}".format(grad[i, j], i, j))
        return np.unravel_index(np.argmax(np.abs(grad)), grad.shape)

    def sample_direction(self, X, determinant_start, residual_start):
        """
        Generates gradients in every basis direction and computes the maximal ascent among those also considering the
        length of the step in that direction. Can be used for debugging purposes

        Parameters
        ----------
        X : Current solution

        determinant_start : current value of the determinant

        residual_start : current value of the residual


        Returns
        -------

        """
        dimension = self.system.n
        detsave = 0
        for i in np.arange(dimension):
            for j in np.arange(i + 1):
                E = np.zeros((dimension, dimension))
                if i == j:
                    E[i, i] = 1
                else:
                    E[i, j] = 1 / np.sqrt(2)
                    E[j, i] = 1 / np.sqrt(2)
                direction = E
                values = np.append(np.logspace(-20., 0), -np.logspace(-20., 0))
                for alpha in values:
                    X1 = X + alpha * direction
                    F, P = self._get_F_and_P(X1)
                    A_F = (self.system.A - self.system.B @ F)
                    residual = self.get_residual(X1, P, F, A_F)
                    determinant = linalg.det(P) * self._get_determinant_R(X1)
                    if determinant > 0:

                        if determinant > determinant_start:

                            if determinant - determinant_start > detsave and linalg.norm(residual) < residual_start:
                                self.logger.info("new best improvement: i,j: {}, {}".format(i, j))
                                self.logger.debug(
                                    "residual: {}\ndeterminant: {}\nalpha: {}".format(linalg.norm(residual),
                                                                                      determinant, alpha))
                                self.logger.info("Improvement!")
                                detsave = determinant - determinant_start

    def _get_H_matrix(self, X):
        """
        Computes the matrix H(X) for given X.

        Parameters
        ----------
        X: Current solution


        Returns
        -------
        H(X)
        """
        raise NotImplementedError

    def _get_F_and_P(self, X):
        """
        Computes current Feedback matrix and Riccati residual

        Parameters
        ----------
        X : Current solution


        Returns
        -------
        F : Current Feedback matrix

        P : Current residual matrix P = Ricc(X)

        """
        raise NotImplementedError

    def riccati_operator(self, X, F=None):
        """
        Computes the residual of the Riccati operator P = Ricc(X). If F is already known, computation effort can be
        saved.

        Parameters
        ----------
        X : Current solution

        F : Current feedback matrix
             (Default value = None)

        Returns
        -------
        P = Ricc(X)
        """
        raise NotImplementedError

    def _get_R(self, X):
        """
        Returns the current value of the 2,2 block of H(X). Is constant if discrete_time == False.
        Parameters
        ----------
        X : Current solution

        Returns
        -------
        [ 0 I]^T  @ H(X) @ [ 0 I ]
        """
        raise NotImplementedError

    def _get_determinant_R(self, X):
        """
        Computes the current determinant H(X), i.e., objective value at given X
        Parameters
        ----------
        X : Current solution


        Returns
        -------
        det(H(X))
        """
        if self._determinant_R is None:
            self._determinant_R = linalg.det(self._get_R(X))
        return self._determinant_R

    def _eig_stable(self):
        """
        Computes whether the eigenvalues of the system matrix A are in the stable region, i.e., in the left-half of the
        complex plane for continuous time systems, or inside the unit disk in the discrete time case.

        Returns
        -------
        Boolean
        """
        raise NotImplementedError

    def _get_res(self, X, P, F, A_F, Delta=None):
        __doc__ = self.get_residual.__doc__
        raise NotImplementedError

    def check_stability(self):
        """Checks whether the system is stable.

         Returns
        -------
        Boolean
        """

        eigs = np.linalg.eig(self.system.A)[0]
        if self._eig_stable(eigs):
            self.logger.info("System is stable")
            return True
        else:
            self.logger.critical("System is not stable. Aborting.")
            raise AnalyticCenterUnstable("System is not stable.")
            return False

    def check_controllability(self):
        """Checks whether the system is controllable.

         Returns
        -------
        Boolean
        """
        gram = self._get_gram()

        if (min(linalg.eigh(gram)[0]) <= 1.e-6):
            self.logger.critical("System is not controllable. Aborting.")
            raise AnalyticCenterUncontrollable("System is not controllable.")
        else:
            self.logger.info("System is controllable.")
            return True

    def check_passivity(self):
        """Checks whether the system is passive.

         Returns
        -------
        Boolean
        """
        try:
            ricc = self.riccati_solver(self.system.A, self.system.B, self.system.Q, self.system.R, self.system.S,
                                       np.identity(self.system.n))

            X = - ricc[0]

            if check_positivity(self._get_H_matrix(X), 'X'):
                self.logger.info("System is passive, if also stable")
                return True
            else:
                self.logger.critical("System is not passive")
                raise AnalyticCenterNotPassive("System is not passive")

        except ValueError as e:
            self.logger.critical(
                "Riccati solver for passivity check did not succeed with message:\n{}".format(e.args[0]))
            raise AnalyticCenterRiccatiSolutionFailed("Riccati solver for passivity check did not succeed")

    def get_residual(self, X, P, F, A_F, Delta=None):
        """ Returns the residual of the gradient of log(det(H(X))).
         If Delta is given, the gradient is computed in direction of Delta. Otherwise, the residual of the corresponding
         linear matrix operator is checked.

        Parameters
        ----------
        X : Current solution

        P : Current residual matrix P = Ricc(X)

        F : Current Feedback matrix

        A_F : = A - B @ F

        Delta : Direction of change, if not None
             (Default value = None)

        Returns
        -------
        norm of the residual
        """
        res = self._get_res(X, P, F, A_F, Delta)
        # self.logger.debug("res: {}".format(res))
        if Delta is None:
            return np.linalg.norm(res)
        else:
            return np.abs(np.real(np.trace(res @ Delta)))

    def characteristics(self, X, Delta=None):
        """
        Computes all characteristic values for a given X.

        Parameters
        ----------
        X : Current solution

        Delta : Direction of change, if not None
             (Default value = None)


        Returns
        -------
        P : Current residual matrix P = Ricc(X)

        F : Current Feedback matrix

        A_F : = A - B @ F

        residual: The current residual of the gradient equation

        determinant: The current determinant of H(X)

        """
        F, P = self._get_F_and_P(X)
        A_F = (self.system.A - self.system.B @ F)
        residual = self.get_residual(X, P, F, A_F, Delta)
        detR = self._get_determinant_R(X)
        R = self._R
        determinant = linalg.det(P) * detR
        self.logger.debug("Current Determinant of R: {}".format(detR))
        return P, R, F, A_F, residual, determinant


class RiccatiOperatorContinuousTime(RiccatiOperator):
    """Child class that defines all the methods of RiccatiOperator class tailored to the continous time"""
    # TODO: Improve performance by saving intermediate results where appropriate
    discrete_time = False
    riccati_solver = staticmethod(control.care)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._determinant_R = None
        self.check_controllability()
        self.check_passivity()

    def _get_H_matrix(self, X):

        A = self.system.A
        B = self.system.B
        H = self.H0 - np.bmat([[A.H @ X + X @ A, X @ B],
                               [B.H @ X, np.zeros((self.system.m, self.system.m))]])
        if self.debug:
            check_positivity(H, "H(X)")
        return H

    def _get_F_and_P(self, X):
        F = np.asmatrix(linalg.solve(self.system.R, self.system.S.H - self.system.B.H @ X))
        P = self.riccati_operator(X, F)
        return F, P

    def _get_res(self, X, P, F, A_F, Delta=None):

        if Delta is None:
            res = P @ A_F
        else:
            res = np.asmatrix(linalg.solve(P, A_F.H))
        res = res + res.H
        return res

    def riccati_operator(self, X, F=None):
        RF = - self.system.B.H @ X + self.system.S.H
        XA = X @ self.system.A
        Ricc = self.system.Q - XA.H - XA
        if F is None:
            Ricc -= symmetric_product_pos_def(F, RF, invertP=True)
        else:
            Ricc -= symmetric_product_pos_def(F, self.system.R)
        return Ricc

    def _get_R(self, X):
        self._R = self.system.R
        return self._R

    def _get_Hamiltonian(self):
        """Computes the associated Hamiltonian matrix and prints the eigenvalues. Used for debugging purposes only.
        """
        A = self.system.A
        B = self.system.B
        Q = self.system.Q
        R = self.system.R
        S = self.system.S
        RinvSH = linalg.solve(R, S.H)
        H1 = A - B @ RinvSH
        Ham = np.bmat([[H1, - B @ linalg.solve(R, B.H)],
                       [-Q + S @ RinvSH, -H1.H]])
        self.logger.debug("Eigenvalues of the Hamiltonian:\n{}".format(linalg.eig(Ham)[0]))

    def _eig_stable(self, eigs):
        return np.max(np.real(eigs)) < 0


    def _get_gram(self):
        return control.gram(control.ss(self.system.A, self.system.B, self.system.C, self.system.D), 'c')

class RiccatiOperatorDiscreteTime(RiccatiOperator):
    """Child class that defines all the methods of RiccatiOperator class tailored to the discrete time"""
    # TODO: Improve performance by saving intermediate results where appropriate
    discrete_time = True
    riccati_solver = staticmethod(control.dare)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._determinant_R = None

    def _get_H_matrix(self, X: np.matrix):
        A = self.system.A
        B = self.system.B
        H = self.H0 - np.bmat([[A.H @ X @ A - X, A.H @ X @ B],
                               [B.H @ X @ A, B.H @ X @ B]])
        if self.debug:
            check_positivity(H, "H(X)")
        return H

    def _get_F_and_P(self, X):
        F = np.asmatrix(linalg.solve(self.system.R - self.system.B.H @ X @ self.system.B,
                                     self.system.S.H - self.system.B.H @ X @ self.system.A))
        P = self.riccati_operator(X, F)
        return F, P

    def _get_res(self, X, P, F, A_F, Delta=None):
        Pinv = linalg.inv(P)
        res = A_F @ Pinv @ A_F.H - Pinv + self.system.B @ linalg.solve(
            self.system.R - self.system.B.H @ X @ self.system.B,
            self.system.B.H)
        return res

    def riccati_operator(self, X, F=None):
        RF = - self.system.B.H @ X @ self.system.A + self.system.S.H
        Ricc = self.system.Q - self.system.A.H @ X @ self.system.A + X
        R = self._get_R(X)
        if F is None:
            Ricc -= RF.H @ linalg.solve(R, RF)
        else:
            Ricc -= F.H @ R @ F
        return Ricc

    def _get_R(self, X):
        self._R = self.system.R - self.system.B.H @ X @ self.system.B
        return self._R

    def _eig_stable(self, eigs):
        return np.max(np.abs(eigs)) < 1


    def _get_gram(self):
        return control.gram(control.ss(self.system.A, self.system.B, self.system.C, self.system.D), 'd')
