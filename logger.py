import logging.config
import numpy as np


def prepare_logger(logging_config):
    logging.config.dictConfig(logging_config)
    np.set_printoptions(linewidth=200)
