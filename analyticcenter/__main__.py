from analyticcenter import get_algorithm_object

from analyticcenter.examples.example2 import sys
import numpy as np

if __name__ == "__main__":

    alg = get_algorithm_object(sys, 'newton', discrete_time=False, save_intermediate=True)
    alg()

