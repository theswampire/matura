import logging
import os
import platform
import sys
from logging.handlers import TimedRotatingFileHandler

from colorama import init, Fore

init(autoreset=True)

LOG_FILE = 'kAI.log'


def get_logger(name: str) -> logging.Logger:
    # from 'https://www.toptal.com/python/in-depth-python-logging'
    # output
    log_format = '[%(asctime)s: %(name)s — %(levelname)s] %(message)s'

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # better to have too much log than not enough

    # console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(Fore.CYAN + log_format))
    logger.addHandler(console_handler)

    # file handler
    if platform.system() != "Windows" or os.getenv("SAVE_LOG_FILE", "False") == "True":
        file_handler = TimedRotatingFileHandler(LOG_FILE, when='midnight', encoding='UTF-8')
        file_handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(file_handler)

    logger.propagate = False  # with this pattern, it's rarely necessary to propagate the error up to parent
    return logger
