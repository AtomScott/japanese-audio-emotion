import logging
import colorlog

def create_logger(level="INFO", name=__name__):
    handler = colorlog.StreamHandler()
    fmt_str = '[%(asctime)s] %(log_color)s%(levelname)s @ line %(lineno)d: %(message)s'
    colors={
		'DEBUG':    'cyan',
		'INFO':     'green',
		'WARNING':  'yellow',
		'ERROR':    'red',
		'CRITICAL': 'red,bg_white',
	}

    handler.setFormatter(
        colorlog.ColoredFormatter(fmt_str,log_colors=colors))

    logger = colorlog.getLogger(name)
    logger.addHandler(handler)
    logger.setLevel(level)

    return logger
