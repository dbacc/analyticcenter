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

import numpy as np
from scipy import linalg

from .direction import DirectionAlgorithm
from ..misc.misc import rsolve


class NewtonDirection(DirectionAlgorithm):
    """ """
    name = "Newton"
    line_search_method = None
    line_search = False
    multi_dimensional = True

    def __init__(self, *args, **kwargs):
        self.lambd = None
        super().__init__(*args, **kwargs)

    def _get_stepzise(self, X, P, F, A, Delta):
        """
        Determines the stepsize that should be used in the Newton step. If lamb > 0.25 a damped Newton method should be
        used.


        Parameters
        ----------
        Delta : The solution of the Newton step
        A : The transformed matrix in the Newton step
        X : Current solution
        P : Current residual matrix P = Ricc(X)
        F : Current Feedback matrix

        Returns
        -------
        step size of the Newton step

        """

        if self.multi_dimensional:

            disc = self.riccati.get_residual(X, P, F, A, Delta)
            if disc < 0:
                self.logger.error("Hessian not pos def!")
                raise ValueError("Hessian not pos def!")
            else:
                lambd = np.sqrt(disc)
                self.lambd = lambd
            if lambd > 0.25:
                self.logger.info("In linearly converging phase")
                # return 1 / (1 + lambd) Scaling really makes things bad atm!
                return 1.
            else:
                self.logger.info("In quadratically converging phase")
                return 1.
        else:
            return 1.

    def _get_direction(self, X0, P0, R0, F0, A_F, fixed_direction=None):
        """
        Computes the next increment Delta_X. If self.line_search is True, a 1d line-search will be performed in addition.

        Parameters
        ----------
        X0 : Current X0

        P0 : Current P0
        R0 : Current R0

        A_F : Current A_F

        fixed_direction : If not None, computation is only done along a 1d subspace spanned by fixed_direction
             (Default value = None)

        Returns
        -------
        Delta_X :  The next increment

        """
        self.fixed_direction = fixed_direction
        A_F_hat, P0_root, S2 = self._transform_system2current_X0(A_F, P0, R0)

        Delta_X_hat = self._newton_step_solver(A_F_hat, S2, P0_root)
        Delta_X = P0_root @ Delta_X_hat @ P0_root

        if self.line_search:
            X0 = Delta_X + X0
            analyticcenter_new, success = self.newton_direction._directional_iterative_algorithm(
                direction_algorithm=self.newton_direction._get_direction, fixed_direction=Delta_X, X0=X0)
            Delta_X += analyticcenter_new.delta_cum
        P0_hat = np.identity(self.system.n)
        alpha = self._get_stepzise(X0, P0, F0, A_F, Delta_X)

        return alpha * Delta_X

    def _transform_system2current_X0(self, A_F, P0, R0):
        """
        Transforms the system to another coordinate system, in which a trivial Delta corresponds to staying at the
        current position and thus formulas are simpler (see paper for details).
        Parameters
        ----------
        A_F : Current A_F

        P0 : Current P0

        R0 : Current R0


        Returns
        -------
        A_F_hat : Transformed A_F

        P0_root : Square root of the current P0

        S2 : Matrix corresponding to the quadratic terms in X_hat of det(H_hat(X_hat))
        """
        P0_root = linalg.sqrtm(P0)
        R0_root = linalg.sqrtm(R0)
        B_hat = P0_root @ self.system.B
        A_F_hat = rsolve(P0_root, (P0_root @ A_F))
        qB, rB = np.linalg.qr(B_hat)
        S2 = np.linalg.solve(R0_root, rB.H) @ qB.H  # Force symmetry!
        S2 = S2.H @ S2
        return A_F_hat, P0_root, S2

    def _splitting_method(self, A, S, P0_root):
        raise NotImplementedError

    def _newton_step_solver(self, A, S, P0_root):
        """
        Computes the Newton step in the transformed coordinate frame.

        Parameters
        ----------
        A : Current A

        S : Matrix corresponding to the quadratic terms in X_hat of det(H_hat(X_hat))

        P0_root : Square root of the current P0


        Returns
        -------
        Delta_X_hat : Newton step in the transformed coordinate frame
        """
        raise NotImplementedError


class NewtonDirectionOneDimensionCT(NewtonDirection):
    """Subclass for computing the Newton step in the one-dimensional continuous-time case"""
    maxiter = 10
    discrete_time = False
    multi_dimensional = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger(self.__class__.__name__)

    # TODO: Check Formula: Slow convergence!!
    def _newton_step_solver(self, A, S, P0_root):
        self.condition = np.linalg.cond(P0_root) ** 2
        search_dir = linalg.solve(P0_root, rsolve(P0_root, self.fixed_direction))
        self.logger.debug("Search direction: {}".format(self.fixed_direction))
        # search_dir = self.fixed_direction
        correction = -(np.real(np.trace(search_dir @ A + A.H @ search_dir)) / (2. * np.real(
            np.trace(search_dir @ S @ search_dir)) + 1. * linalg.norm(
            search_dir @ A + A.H @ search_dir) ** 2))
        return correction * search_dir

class NewtonDirectionOneDimensionDT(NewtonDirection):
    """Subclass for computing the Newton step in the one-dimensional discrete-time case"""
    maxiter = 50
    discrete_time = True
    multi_dimensional = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger(self.__class__.__name__)

    def _newton_step_solver(self, A, S, P0_root):
        self.condition = np.linalg.cond(P0_root)
        n = self.system.n
        identity = np.identity(n)
        AAH = A @ A.H
        search_dir = linalg.solve(P0_root, rsolve(P0_root, self.fixed_direction))
        second_order = - S @ search_dir @ S @ search_dir - 2 * AAH @ search_dir @ S @ search_dir - AAH @ search_dir @ AAH @ search_dir + 2 * A @ search_dir @ A.H @ search_dir - search_dir @ search_dir
        self.logger.debug("Search direction: {}".format(self.fixed_direction))
        return -0.5 * (np.real(np.trace(search_dir @ (identity - S - AAH))) / (np.real(
            np.trace(second_order)))) * search_dir


class NewtonDirectionMultipleDimensions(NewtonDirection):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _stopping_criterion(self):

        if self.lambd is not None and self.lambd ** 2 < self.abs_tol:
            return True
        else:
            return False


class NewtonDirectionMultipleDimensionsCT(NewtonDirectionMultipleDimensions):
    """Subclass for computing the Newton step in the multi-dimensional continuous-time case"""


    name = "NewtonMDCT"
    line_search_method = NewtonDirectionOneDimensionCT
    line_search = False
    discrete_time = False


    def __init__(self, *args, **kwargs):
        self.newton_direction = self.line_search_method(*args, **kwargs)
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger(self.__class__.__name__)


    def _newton_step_solver(self, A, S, P0_root):
        # TODO: use Schur Form
        rhs = -np.ravel(A + A.H)
        AAH = A @ A.H
        n = self.system.n
        identity = np.identity(n)
        lhs = np.kron(A.T, A) + np.kron(np.conj(A), A.H) + np.kron(identity, AAH) + np.kron(AAH.T,
                                                                                            identity) + np.kron(
            identity, S) + np.kron(S, identity)


        self.logger.debug("Current lhs and rhs\n{}\n{}".format(lhs, rhs))
        Delta = np.asmatrix(linalg.solve(lhs, rhs, assume_a='pos'))
        self.logger.debug("Solution Delta:\n{}".format(Delta))
        Delta = np.reshape(Delta, [n, n])
        self.logger.debug("Reshaped Delta:\n{}".format(Delta))
        Delta = 0.5 * (Delta + Delta.H)
        self.condition = np.linalg.cond(lhs)
        # check if indeed solution:

        if __debug__:
            self.logger.debug("Condition Number of the Hessian: {}".format(np.linalg.cond(lhs)))
            self._check(A, S, Delta)

        return Delta


    def _check(self, A, S, Delta):
        """Convenience function for debugging, whether Newton equation is solved accurately."""
        AAH = A @ A.H
        n = self.system.n
        identity = np.identity(n)
        res = - A @ Delta @ A - AAH @ Delta - A.H @ Delta @ A.H - Delta @ AAH.H - S @ Delta - Delta @ S - A - A.H
        self.logger.debug("norm of the residual: {}".format(linalg.norm(res)))
        det_factor = linalg.det(identity - Delta @ A - A.H @ Delta - Delta @ S @ Delta)
        if det_factor < 1.:
            self.logger.warning("det factor by newton step is less than 1: {}".format(det_factor))
        else:
            self.logger.debug("det factor by newton step: {}".format(det_factor))


class NewtonDirectionIterativeCT(NewtonDirectionMultipleDimensionsCT):
    """Subclass for computing the Newton step in the multi-dimensional continuous-time case with an iterative algorithm"""
    discrete_time = False
    maxiter_newton = 200
    tol_newton = 10 ** -10

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger(self.__class__.__name__)

    def _newton_step_solver(self, A, S, P0_root):
        alpha = 0  # TODO how to choose alpha?
        n = self.system.n
        identity = np.identity(n)
        AAH = A @ A.H
        API = A + identity
        Delta = 0 * A
        i = 0
        test = self.tol_newton + 1
        while i < self.maxiter_newton and linalg.norm(test) > self.tol_newton:
            rhs = 2 * alpha * Delta - A @ Delta @ A - A.H @ Delta @ A.H - A - A.H
            Delta = linalg.solve_lyapunov(AAH + S + alpha * identity, rhs)

            test = - A @ Delta @ A - AAH @ Delta - A.H @ Delta @ A.H - Delta @ AAH.H - S @ Delta - Delta @ S - A - A.H
            i += 1
            self.logger.debug(
                "Current value of Delta in Newton iteration:\n{}\nresidual: {}".format(Delta, linalg.norm(test)))
        return Delta


class NewtonDirectionMultipleDimensionsDT(NewtonDirectionMultipleDimensions):
    """Subclass for computing the Newton step in the multi-dimensional discrete-time case"""
    discrete_time = True
    line_search_method = NewtonDirectionOneDimensionDT

    def __init__(self, *args, **kwargs):
        self.newton_direction = self.line_search_method(*args, **kwargs)
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger(self.__class__.__name__)

    def _newton_step_solver(self, A, S, P0_root):
        # TODO: use Schur Form

        AAH = A @ A.H
        n = self.system.n
        identity = np.identity(n)
        rhs = np.ravel(identity - S - AAH)
        lhs = np.kron(S, S) + np.kron(S, AAH) + np.kron(AAH.T, S) + np.kron(AAH.T, AAH) - np.kron(np.conj(A), A) \
              - np.kron(A.T, A.H) + np.kron(identity, identity)
        self.logger.debug("Current lhs and rhs\n{}\n{}".format(lhs, rhs))
        Delta = linalg.solve(lhs, rhs)
        self.logger.debug("Solution Delta:\n{}".format(Delta))
        Delta = np.reshape(Delta, [n, n])
        self.logger.debug("Reshaped Delta:\n{}".format(Delta))
        self.condition = np.linalg.cond(lhs)
        # check if indeed solution:
        # ipdb.set_trace()
        if self.debug:
            self._check(A, S, Delta)
        return Delta

    def _check(self, A, S, Delta):
        n = self.system.n
        identity = np.identity(n)
        AAH = A @ A.H
        res = S @ Delta @ S + AAH @ Delta @ S + S @ Delta @ AAH + AAH @ Delta @ AAH - A @ Delta @ A.H - A.H @ Delta @ A + Delta - identity + S + AAH
        self.logger.debug("norm of the residual: {}".format(linalg.norm(res)))

        det_factor = linalg.det(np.bmat([[identity - S @ Delta, A], [A.H @ Delta, identity + Delta]]))
        if det_factor < 1- 1.e-14:
            self.logger.critical("det factor by newton step is less than 1: {}".format(det_factor))
        else:
            self.logger.debug("det factor by newton step: {}".format(det_factor))

