import inspect
import os

import yaml

from analyticcenter.algorithm import get_analytic_center_object
from analyticcenter.direction import NewtonDirectionMultipleDimensionsCT

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
    from examples.example4 import sys
    alg = get_analytic_center_object(sys, 10 ** (-8), discrete_time=False)
    direction_method = NewtonDirectionMultipleDimensionsCT()
    direction_method()
