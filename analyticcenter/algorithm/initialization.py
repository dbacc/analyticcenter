##
# Copyright (c) 2019
##
# @author: Daniel Bankmann
# @company: Technische Universität Berlin
##
# This file is part of the python package analyticcenter
# (see https://gitlab.tu-berlin.de/PassivityRadius/analyticcenter/)
##
# License: 3-clause BSD, see https://opensource.org/licenses/BSD-3-Clause
##
import logging

import control
from ..misc.control import dare as mydare
from ..misc.control import care as mycare
import numpy as np
from scipy import linalg

from .direction import DirectionAlgorithm
from .newton import NewtonDirectionOneDimensionCT, NewtonDirectionOneDimensionDT


class InitialX(DirectionAlgorithm):
    """Computation of Initial solution that is strictly positive for the LMI"""
    maxiter = 100

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, force_bisection=False):
        """
        Computation of an initial strictly positive solution of the LMI.
        There are several heuristics that lead to interior points and one 'save' approach.
        Here, we first compute the geometric mean (heuristic approach).
        We first compute two different solutions of the Riccati equation (stabilizing and
        anti-stabilizing).
        Then we hope, that combining them in a clever way (geometric mean) leads us to a point
        that is in the interior.
        We also try the bisection approach, which in theory, brings us to a point in the interior.
        However, in the examples we tried, the solutions produced by the geometric mean approach
        led to better initial guesses.

        Parameters
        -------

        force_bisection: Boolean
        If set to true, the solution of the bisection will be taken, even if the geometric_mean
        approach led to a better initial guess.

        Returns
        -------
        Xinit : Initial solution

        success : Boolean
        """
        self.logger.info('Computing initial X')

        X0_geom = self._geometric_mean()
        self.logger.debug(
            "Eigenvalues of X_init_guess: {}".format(
                linalg.eigh(X0_geom)[0]))
        H_geom = self.riccati._get_H_matrix(X0_geom)
        mineig_geometric_mean = linalg.eigh(H_geom)[0][0]
        det_geom = linalg.det(H_geom)
        self.logger.info("Computed initial guess with geometric mean approach.\n"
                         "det(H(X0)) = {}".format(det_geom))

        self.logger.debug(
            "Eigenvalues of H(X_init_guess): {}".format(
                linalg.eigh(self.riccati._get_H_matrix(X0_geom))[0]))

        X0_bisect = self._bisection_approach()
        H_bisect = self.riccati._get_H_matrix(X0_bisect)
        det_bisect = linalg.det(H_bisect)
        mineig_bisect = linalg.norm(H_bisect, -2)
        self.logger.info("Computed initial guess with bisection approach.\n"
                         "det(H(X0)) = {}".format(det_bisect))

        if det_bisect > det_geom or force_bisection and mineig_bisect > 0:
            self.logger.info("Taking solution computed with bisection approach")
            X0 = X0_bisect
            success = True
        elif mineig_geometric_mean > 0:
            self.logger.info("Taking solution computed with geometric mean approach")
            X0 = X0_geom
            success = True
        else:
            success = False
            X0 = np.array([])
        return X0, success

    def _geometric_mean(self):
        X_plus = -np.asmatrix(self.riccati_solver(self.system.A,
                                                  self.system.B,
                                                  self.system.Q,
                                                  self.system.R,
                                                  self.system.S,
                                                  np.identity(self.system.n),
                                                  True))
        Am = -self.system.A
        Bm = self.system.B
        Sm = self.system.S
        Qm = -self.system.Q
        Rm = -self.system.R
        X_minus = -np.asmatrix(self.riccati_solver(self.system.A,
                                                  self.system.B,
                                                  self.system.Q,
                                                  self.system.R,
                                                  self.system.S,
                                                  np.identity(self.system.n),
                                                  False))


        if np.isclose(linalg.norm(X_plus - X_minus), 0):
            self.logger.critical(
                "X_+ and X_- are (almost) identical: No interior!")
        self.logger.debug(
            "Eigenvalues of X_plus: {}".format(
                linalg.eigh(X_plus)[0]))


        self.logger.debug("Eigenvalues of H(X_plus): {}".format(
            linalg.eigh(self.riccati._get_H_matrix(X_plus))[0]))
        self.logger.debug(
            "Eigenvalues of X_minus: {}".format(
                linalg.eigh(X_minus)[0]))
        self.logger.debug("Eigenvalues of H(X_minus): {}".format(
            linalg.eigh(self.riccati._get_H_matrix(X_minus))[0]))
        X0_geom = X_minus @ linalg.sqrtm(linalg.solve(X_minus, X_plus))
        return X0_geom

    def _bisection_approach(self):
        max_pert = min(0.5, np.linalg.norm(self.system.R, -2))
        upper_bound = max_pert
        lower_bound = 0
        maxiter = 20
        for i in range(maxiter):
            pert = lower_bound + 0.5*(upper_bound - lower_bound)

            self.logger.debug(
                "upper_bound: {}\nlower_bound: {}\npert: {}".format(
                    upper_bound, lower_bound, pert))
            Apert, Bpert, Qpert, Rpert, Spert = self._perturbed_system_matrices(
                pert)

            try:
                X_plus_int = -np.asmatrix(self.riccati_solver(Apert, Bpert,
                                                              Qpert, Rpert,
                                                              Spert,
                                                              np.identity(self.system.n)))
                mineig = np.linalg.eigh(self.riccati._get_H_matrix(X_plus_int))[0][0]
            except ValueError:
                self.logger.debug(
                    '''Perturbation pert = {} too big,
                    Riccati Equation does not have a solution'''.format(pert))
                upper_bound = pert
                continue
            if mineig > 0:
                lower_bound = pert

            else:
                upper_bound = upper_bound/2
        return X_plus_int


class InitialXCT(InitialX):
    line_search_method = NewtonDirectionOneDimensionCT
    riccati_solver = staticmethod(lambda *args: mycare(*args)[0])

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.newton_direction = self.line_search_method(*args, **kwargs)
        self.logger = logging.getLogger(self.__class__.__name__)

    def _perturbed_system_matrices(self, pert):
        Apert = self.system.A + pert * np.identity(self.system.n)
        Bpert = self.system.B
        Qpert = self.system.Q
        Rpert = self.system.R - 2 * pert * np.identity(self.system.m)
        Spert = self.system.S
        return Apert, Bpert, Qpert, Rpert, Spert


class InitialXDT(InitialX):
    line_search_method = NewtonDirectionOneDimensionDT
    riccati_solver = staticmethod(lambda *args: mydare(*args)[0])

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.newton_direction = self.line_search_method(*args, **kwargs)
        self.logger = logging.getLogger(self.__class__.__name__)

    def _perturbed_system_matrices(self, pert):
        factor = 1 - 2 * pert
        sqfac = np.sqrt(factor)
        Apert = self.system.A / sqfac
        Bpert = self.system.B / sqfac
        Qpert = self.system.Q / factor
        Rpert = self.system.R - 2 * pert * np.identity(self.system.m)
        Rpert = Rpert / factor
        Spert = self.system.S / factor
        return Apert, Bpert, Qpert, Rpert, Spert
