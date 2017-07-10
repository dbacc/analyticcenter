import numpy as np
from scipy import linalg

from misc.misc import schur_complement, rsolve
import logging
import ipdb
from functools import partial


class AnalyticCenter(object):
    """ToDo"""
    __debug = True

    def __init__(self, system, tol):
        self.system = system
        self.X0 = None
        self.H0 = None
        self.H = None
        self.tol = tol
        self.maxiter = 2000
        self.__init_H0()
        self.debug = True
        self.logger = logging.getLogger(__name__)
        self.largest_eigenvalues = np.array([])
        self.smallest_eigenvalues = np.array([])

    def __init_H0(self):
        self.H0 = np.bmat([[self.system.Q, self.system.S], [self.system.S.transpose(), self.system.R]])

    def _get_H_matrix(self, X: np.matrix):
        A = self.system.A
        B = self.system.B
        H = self.H0 - np.bmat([[A.T @ X + X @ A, X @ B],
                               [B.T @ X, np.zeros((self.system.m, self.system.m))]])
        return H

    def _get_Delta_H(self, Delta_X):
        return self._get_H_matrix(Delta_X) - self.H0

    def _get_initial_X(self):
        if self.X0 is None:
            self.logger.info('Computing initial X')
            X_plus = linalg.solve_continuous_are(self.system.A, self.system.B, self.system.Q, self.system.R)
            Am = -self.system.A
            Bm = -self.system.B
            X_minus = linalg.solve_continuous_are(Am, Bm, self.system.Q, self.system.R)
            self.search_direction = (X_minus + X_plus)
            self.X0 = -0.5 * self.search_direction  # We use negative definite notion of solutions for Riccati equation
            self.logger.info("Improving Initial X with Newton approach")
            Xinit = self._directional_iterative_algorithm(
                direction=partial(self._get_newton_direction, newton_step_solver=self._solve_newton_step_1d))

        else:
            self.logger.info("initial X is already set")
            Xinit = self.X0
        return Xinit

    def _get_F_and_P(self, X):
        F = linalg.solve(self.system.R, self.system.S.H - self.system.B.H @ X)
        P = self.riccati_operator(X)
        return F, P

    def _transform_system2current_X0(self, A_F, P0):
        P0_root = linalg.sqrtm(P0)
        B_hat = P0_root @ self.system.B
        A_F_hat = rsolve(P0_root, (P0_root @ A_F))
        S2 = B_hat @ linalg.solve(self.system.R, B_hat.H)  # only works in continuous time
        return A_F_hat, P0_root, S2

    def _directional_iterative_algorithm(self, direction):
        def print_information(steps_count, residual, determinant, X):
            self.logger.info("Current step: {}\tResidual: {}".format(steps_count, residual))
            self.logger.debug("Current objective value (det(H(X))): {}".format(determinant))
            self.logger.debug("Current X:\n{}".format(X))
            if np.real(determinant) < 0:
                self.logger.critical("Something went wrong. Determinant ist negative. Aborting...")
                raise ValueError("Something went wrong. Determinant ist negative")

        determinant_R = linalg.det(self.system.R)
        # noinspection PyPep8Naming
        X = self._get_initial_X()
        self.logger.debug("Initial X:\n{}".format(X))
        F, P = self._get_F_and_P(X)
        determinant = linalg.det(P) * determinant_R
        A_F = (self.system.A - self.system.B @ F)
        steps_count = 0
        residual = self.get_residual(X, P, F, A_F)
        Delta_residual = float("inf")
        while residual > self.tol and Delta_residual > self.tol and steps_count < self.maxiter:
            # ipdb.set_trace()

            print_information(steps_count, residual, determinant, X)
            Delta_X = direction(X, P, A_F)
            Delta_residual = linalg.norm(Delta_X)
            X = X + Delta_X
            self.logger.debug("Updating current X by Delta:_X:\n{}".format(Delta_X))

            F, P = self._get_F_and_P(X)
            A_F = (self.system.A - self.system.B @ F)
            residual = self.get_residual(X, P, F, A_F)
            determinant = linalg.det(P) * determinant_R
            steps_count += 1
        print_information(steps_count, residual, determinant, X)
        return X

    def newton(self):
        self.logger.info("Computing Analytic Center with Newton approach")
        self._directional_iterative_algorithm(
            direction=partial(self._get_newton_direction, newton_step_solver=self._solve_newton_step_nd))

    def _get_newton_direction(self, X0, P0, A_F, newton_step_solver):
        A_F_hat, P0_root, S2 = self._transform_system2current_X0(A_F, P0)

        Delta_X_hat = newton_step_solver(A_F_hat, S2, P0_root)
        Delta_X = P0_root @ Delta_X_hat @ P0_root
        return Delta_X

    def _solve_newton_step_nd(self, A, S, P0_root):
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
        if self.debug:
            res = - A @ Delta @ A - AAH @ Delta - A.H @ Delta @ A.H - Delta @ AAH.H - S @ Delta - Delta @ S - A - A.H
            self.logger.debug("norm of the residual: {}".format(linalg.norm(res)))
            det_factor = linalg.det(identity - Delta @ A - A.H @ Delta - Delta @ S @ Delta)
            if det_factor < 1.:
                self.logger.critical("det factor by newton step is less than 1: {}".format(det_factor))
            else:
                self.logger.debug("det factor by newton step: {}".format(det_factor))
        return Delta

    def _solve_newton_step_1d(self, A, S, P0_root):
        search_dir = linalg.solve(P0_root, rsolve(P0_root, self.search_direction))
        self.logger.debug("Search direction: {}".format(self.search_direction))
        return -0.5 * (np.real(np.trace(search_dir @ A + A.H @ search_dir)) / np.real(
            np.trace(search_dir @ S @ search_dir))) * search_dir

    def steepest_ascent(self):
        self.logger.info("Computing Analytic Center with steepest_ascent approach")
        self._directional_iterative_algorithm(direction=self._get_ascent_direction_direct)

    def get_residual(self, X, P, F, A_F):
        res1 = -self.system.B.T @ X + self.system.S.T - self.system.R @ F
        res2 = self.system.Q - self.system.A.T @ X - X @ self.system.A - F.T @ self.system.R @ F - P
        res3 = P @ A_F
        res3 = res3 + res3.T
        self.logger.debug("\nres1:\n{},\nres2: {},\nres3: {}".format(res1, res2, res3))
        return np.linalg.norm(linalg.block_diag(res1, res2, res3))

    def _get_ascent_direction_direct(self, X, A_F):
        A_T = rsolve(self.riccati_operator(X), A_F)

        self.logger.debug("Current Feedback Matrix A_F:\n{}".format(A_F))
        self.logger.debug("Current Feedback Matrix transformed A_T:\n{}".format(A_T))
        A_T_symmetric = A_T + np.asmatrix(A_T).H
        # We're assuming simple eigenvalues here!
        largest_eigenpair = linalg.eigh(A_T_symmetric, eigvals=(self.system.n - 1, self.system.n - 1))
        smallest_eigenpair = linalg.eigh(A_T_symmetric, eigvals=(0, 0))
        self.logger.debug("Symmetric part of A_T:\n{}".format(A_T_symmetric))
        largest_eigenvector = largest_eigenpair[1]
        largest_eigenvalue = largest_eigenpair[0]

        smallest_eigenvector = smallest_eigenpair[1]
        smallest_eigenvalue = smallest_eigenpair[0]

        if np.abs(smallest_eigenvalue) < np.abs(largest_eigenvalue):
            largest_abs_eigenvalue = largest_eigenvalue
            smallest_abs_eigenvalue = smallest_eigenvalue
            largest_abs_eigenvector = largest_eigenvector
        else:
            largest_abs_eigenvalue = smallest_eigenvalue
            largest_abs_eigenvector = smallest_eigenvector
            smallest_abs_eigenvalue = largest_eigenvalue
        self.largest_eigenvalues = np.append(self.largest_eigenvalues, largest_abs_eigenvalue)
        self.smallest_eigenvalues = np.append(self.smallest_eigenvalues, smallest_abs_eigenvalue)
        Delta_X = largest_abs_eigenvector @ np.asmatrix(largest_abs_eigenvector).H

        self.logger.debug(
            "largest eigenvalue: {},\tcorresponding eigenvector: {},\tnorm: {}".format(largest_eigenvalue,
                                                                                       largest_eigenvector,
                                                                                       linalg.norm(
                                                                                           largest_eigenvector)))
        self.logger.debug(
            "smallest eigenvalue: {},\tcorresponding eigenvector: {},\tnorm: {}".format(smallest_eigenvalue,
                                                                                        smallest_eigenvector,
                                                                                        linalg.norm(
                                                                                            smallest_eigenvector)))
        # if self.debug:
        #     self.det_direction_plot(X, Delta_X)
        # ipdb.set_trace()
        stepsize = self._get_ascent_step_size(X, largest_abs_eigenvector)
        return stepsize * Delta_X

    def _get_ascent_direction_Ttransformed(self, X, A_F):
        T = linalg.sqrtm(self.riccati_operator(X))
        A_T = T @ rsolve(T, A_F)

        self.logger.debug("Current Feedback Matrix A_F:\n{}".format(A_F))
        self.logger.debug("Current Feedback Matrix transformed A_T:\n{}".format(A_T))

        A_T_symmetric = A_T + np.asmatrix(A_T).H
        # We're assuming simple eigenvalues here!
        largest_eigenpair = linalg.eigh(A_T_symmetric, eigvals=(self.system.n - 1, self.system.n - 1))
        smallest_eigenpair = linalg.eigh(A_T_symmetric, eigvals=(0, 0))

        self.logger.debug("Symmetric part of A_T:\n{}".format(A_T_symmetric))
        largest_eigenvector = largest_eigenpair[1]
        largest_eigenvalue = largest_eigenpair[0]
        smallest_eigenvector = smallest_eigenpair[1]
        smallest_eigenvalue = smallest_eigenpair[0]

        if np.abs(smallest_eigenvalue) < np.abs(largest_eigenvalue):
            largest_abs_eigenvalue = largest_eigenvalue
            largest_abs_eigenvector = largest_eigenvector
        else:
            largest_abs_eigenvalue = smallest_eigenvalue
            largest_abs_eigenvector = smallest_eigenvector
        Delta_X = T @ largest_abs_eigenvector @ np.asmatrix(largest_abs_eigenvector).H @ T

        self.logger.debug(
            "largest eigenvalue: {},\tcorresponding eigenvector: {},\tnorm: {}".format(largest_eigenvalue,
                                                                                       largest_eigenvector,
                                                                                       linalg.norm(
                                                                                           largest_eigenvector)))
        self.logger.debug(
            "smallest eigenvalue: {},\tcorresponding eigenvector: {},\tnorm: {}".format(smallest_eigenvalue,
                                                                                        smallest_eigenvector,
                                                                                        linalg.norm(
                                                                                            smallest_eigenvector)))
        stepsize = self._get_ascent_step_size(X, T @ largest_abs_eigenvector)
        return stepsize * Delta_X

    def _get_ascent_step_size(self, X, eigenvector):
        # perm = np.matrix([[0, 1], [1, 0]])
        H = self._get_H_matrix(X)
        eigenvector = np.asmatrix(eigenvector)  # / linalg.norm(eigenvector)
        self.logger.debug("H(X): {}".format(H))
        ProjectionMatrix = np.asmatrix(linalg.block_diag(eigenvector, eigenvector)).H @ np.bmat(
            [[self.system.A, self.system.B],
             [np.identity(self.system.n), np.zeros((self.system.n, self.system.m))]])

        ProjectedMatrix = ProjectionMatrix @ linalg.solve(H, ProjectionMatrix.H)
        c = ProjectedMatrix[1, 0]
        ab = ProjectedMatrix[0, 0] * ProjectedMatrix[1, 1]
        stepsize = c / (-ab + c ** 2)
        stepsize = c / (-ab + c ** 2)
        # ipdb.set_trace()
        self.logger.debug("Chosen stepsize: {}".format(stepsize))
        return stepsize

    def riccati_operator(self, X):
        RF = - self.system.B.T @ X + self.system.S.T
        Ricc = self.system.Q - self.system.A.T @ X - X @ self.system.A - RF.T @ linalg.solve(self.system.R, RF)
        return Ricc

    def det_direction_plot(self, X, Delta_X):
        self.logger.debug("Creating det direction plot...")
        alpha = np.linspace(-1., 1., 10000)
        det_alpha = [linalg.det(self.riccati_operator(X + val * Delta_X)) for val in alpha]
        # ipdb.set_trace()
        plt.plot(alpha, det_alpha)
        self.logger.debug("Maximum reached at alpha = {}".format(alpha[np.argmax(det_alpha)]))
        # plt.show()
        return alpha[np.argmax(det_alpha)]

    def gradient_sweep(self, X):
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
                self.logger.debug("Gradient: {} Direction: {}, {}".format(grad[i, j], i, j))
        return np.unravel_index(np.argmax(np.abs(grad)), grad.shape)
