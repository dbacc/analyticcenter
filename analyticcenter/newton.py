import logging

import ipdb
import numpy as np
from scipy import linalg

from analyticcenter.direction import DirectionAlgorithm
from misc.misc import rsolve


class NewtonDirection(DirectionAlgorithm):
    name = "Newton"

    def __init__(self):
        raise NotImplementedError("Should not be called from Base class")

    def _get_direction(self, X0, P0, R0, A_F, fixed_direction=None):
        self.fixed_direction = fixed_direction
        A_F_hat, P0_root, S2 = self._transform_system2current_X0(A_F, P0, R0)

        Delta_X_hat = self._newton_step_solver(A_F_hat, S2, P0_root)
        # ipdb.set_trace()
        Delta_X = P0_root @ Delta_X_hat @ P0_root
        return Delta_X

    def _transform_system2current_X0(self, A_F, P0, R0):
        P0_root = linalg.sqrtm(P0)
        B_hat = P0_root @ self.system.B
        A_F_hat = rsolve(P0_root, (P0_root @ A_F))
        S2 = B_hat @ linalg.solve(R0, B_hat.H)  # only works in continuous time
        return A_F_hat, P0_root, S2

    def _splitting_method(self, A, S, P0_root):
        raise NotImplementedError


class NewtonDirectionMultipleDimensionsCT(NewtonDirection):
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)


    def _newton_step_solver(self, A, S, P0_root):
        '''solve newton step using kronecker products. Will be way too expensive in general!'''
        # TODO: use Schur Form
        rhs = np.ravel(A + A.H)
        AAH = A @ A.H
        n = self.system.n
        identity = np.identity(n)
        lhs = - np.kron(A.T, A) - np.kron(np.conj(A), A.H) - np.kron(identity, AAH) - np.kron(AAH.T,
                                                                                              identity) - np.kron(
            identity, S) - np.kron(S, identity)

        self.logger.debug("Current lhs and rhs\n{}\n{}".format(lhs, rhs))
        Delta = linalg.solve(lhs, rhs)
        self.logger.debug("Solution Delta:\n{}".format(Delta))
        Delta = np.reshape(Delta, [n, n])
        self.logger.debug("Reshaped Delta:\n{}".format(Delta))

        # check if indeed solution:
        # ipdb.set_trace()


        if self.debug:
            self._check(A, S, Delta)

        return Delta

    def _check(self, A, S, Delta):
        AAH = A @ A.H
        n = self.system.n
        identity = np.identity(n)
        res = - A @ Delta @ A - AAH @ Delta - A.H @ Delta @ A.H - Delta @ AAH.H - S @ Delta - Delta @ S - A - A.H
        self.logger.debug("norm of the residual: {}".format(linalg.norm(res)))
        det_factor = linalg.det(identity - Delta @ A - A.H @ Delta - Delta @ S @ Delta)
        if det_factor < 1.:
            self.logger.critical("det factor by newton step is less than 1: {}".format(det_factor))
        else:
            self.logger.debug("det factor by newton step: {}".format(det_factor))


class NewtonDirectionIterativeCT(NewtonDirectionMultipleDimensionsCT):
    maxiter_newton = 200
    tol_newton = 10 ** -10

    def __init__(self):
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


class NewtonDirectionMultipleDimensionsDT(NewtonDirection):
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def _newton_step_solver(self, A, S, P0_root):
        '''solve newton step using kronecker products. Will be way too expensive in general!'''
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
        if det_factor < 1.:
            self.logger.critical("det factor by newton step is less than 1: {}".format(det_factor))
        else:
            self.logger.debug("det factor by newton step: {}".format(det_factor))


class NewtonDirectionOneDimensionCT(NewtonDirection):
    fixed_direction = None
    maxiter = 5


    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    # TODO: Check Formula: Slow convergence!!
    def _newton_step_solver(self, A, S, P0_root):
        search_dir = linalg.solve(P0_root, rsolve(P0_root, self.fixed_direction))
        self.logger.debug("Search direction: {}".format(self.fixed_direction))
        # search_dir = self.fixed_direction
        correction = -(np.real(np.trace(search_dir @ A + A.H @ search_dir)) / (2.*np.real(
            np.trace(search_dir @ S @ search_dir)) + 1. * linalg.norm(
            search_dir @ A + A.H @ search_dir) ** 2))
        # if np.isclose(correction,0.):
        #     ipdb.set_trace()
        return correction * search_dir


class NewtonDirectionOneDimensionDT(NewtonDirection):
    fixed_direction = None
    maxiter = 5

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    # TODO: Check Formula: Slow convergence!!
    def _newton_step_solver(self, A, S, P0_root):
        n = self.system.n
        identity = np.identity(n)
        AAH = A @ A.H
        search_dir = linalg.solve(P0_root, rsolve(P0_root, self.fixed_direction))
        second_order = - S @ search_dir @ S @ search_dir - 2 * AAH @ search_dir @ S @ search_dir - AAH @ search_dir @ AAH @ search_dir + 2 * A @ search_dir @ A.H @ search_dir - search_dir @ search_dir
        self.logger.debug("Search direction: {}".format(self.fixed_direction))
        return -0.5 * (np.real(np.trace(search_dir @ (identity - S - AAH))) / (np.real(
            np.trace(second_order)))) * search_dir