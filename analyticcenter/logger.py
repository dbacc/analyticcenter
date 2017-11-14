import logging.config
import numpy as np
import inspect
import os
import yaml


def load_config():
    cmd_folder = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile(inspect.currentframe()))[0]))
    configpath = os.path.join(cmd_folder, 'config/config.yaml')
    with open(configpath, 'rt') as f:
        config = yaml.safe_load(f.read())
    return config


def prepare_logger(logging_config):
    logging.config.dictConfig(logging_config)
    np.set_printoptions(linewidth=200)


logging_config = load_config()
prepare_logger(logging_config)