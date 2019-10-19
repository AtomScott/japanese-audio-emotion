import logging
import colorlog

def create_logger(level="INFO", name=__name__):
    handler = colorlog.StreamHandler()
    fmt_str = '[%(asctime)s] %(log_color)s%(levelname)s @ line %(lineno)d: %(message)s'
    handler.setFormatter(colorlog.ColoredFormatter(fmt_str))

    logger = colorlog.getLogger(name)
    logger.addHandler(handler)
    logger.setLevel(level)

    return logger
