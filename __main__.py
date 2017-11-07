import inspect
import os

import yaml

from analyticcenter.algorithm import get_algorithm_object
from analyticcenter.newton import NewtonDirectionMultipleDimensionsCT
from analyticcenter.steepestascent import SteepestAscentDirectionCT
from analyticcenter.diagonal import DiagonalDirection
from logger import prepare_logger


# from examples.example3 import sys

def load_config():
    cmd_folder = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile(inspect.currentframe()))[0]))
    configpath = os.path.join(cmd_folder, 'config/config.yaml')
    with open(configpath, 'rt') as f:
        config = yaml.safe_load(f.read())
    return config


if __name__ == "__main__":
    logging_config = load_config()
    prepare_logger(logging_config)
    from examples.rlc import sys
    alg = get_algorithm_object(sys, discrete_time=False, save_intermediate=False, abs_tol=9e-8)
    direction_method1 = DiagonalDirection()
    direction_method1.maxiter = 1000
    (ac, success) = direction_method1()
