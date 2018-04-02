import logging


def init_console_logger(logger, verbose=False):
    """
    Initializes logging to stdout
    Args:
        logger:  Logger object
                 (Type: logging.Logger)
    Kwargs:
        verbose:  If true, prints verbose information to stdout. False by default.
                  (Type: bool)
    """
    # Log to stderr also
    stream_handler = logging.StreamHandler()
    if verbose:
        stream_handler.setLevel(logging.DEBUG)
    else:
        stream_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)