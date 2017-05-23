import logging.config

def prepare_logger(logging_config):
    logging.config.dictConfig(logging_config)