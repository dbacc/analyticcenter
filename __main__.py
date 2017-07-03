import inspect
import os

import numpy as np
import yaml




from logger import prepare_logger

from analyticcenter.algorithm import AnalyticCenter
from analyticcenter.linearsystem import OptimalControlSystem


def load_config():
    cmd_folder = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile(inspect.currentframe()))[0]))
    configpath = os.path.join(cmd_folder, 'config/config.yaml')
    with open(configpath, 'rt') as f:
        config = yaml.safe_load(f.read())
    return config





def init_example1():
    A = np.matrix([[1, -1], [1, 1]])
    B = np.matrix([[1], [1]])
    C = B.T
    D = C @ B
    Q = np.identity(2)
    sys = OptimalControlSystem(A, B, C, D, Q, 0 * B, 2 * D)
    return sys


def init_example2():
    sysmat = np.load('example-n-4-m-2.npy')
    sys = OptimalControlSystem(*sysmat)
    return sys





if __name__ == "__main__":
    logging_config = load_config()
    prepare_logger(logging_config)
    sys = init_example1()

    alg = AnalyticCenter(sys, 10 ** (-3))
