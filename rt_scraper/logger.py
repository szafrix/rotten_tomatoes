import os
import datetime as dt
import logging
import logging.config
from logging.handlers import RotatingFileHandler


class ColoredFormatter(logging.Formatter):
    bright_red_bckg = "\x1b[91m"
    red_bckg = "\x1b[31m"
    yellow_bckg = "\x1b[33m"
    green_bckg = "\x1b[32m"
    reset = "\x1b[0m"
    format = "%(asctime)s - %(levelname)s - %(message)s"

    FORMATS = {
        logging.INFO: green_bckg + format + reset,
        logging.WARNING: yellow_bckg + format + reset,
        logging.ERROR: red_bckg + format + reset,
        logging.CRITICAL: bright_red_bckg + format + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


logging_config = {
    "version": 1,
    "disable_existing_loggers": True,
    "loggers": {
        "scraper_log": {
            "level": "INFO",
        }
    },
}


def get_scraper_logger():
    if not os.path.exists("rt_scraper/logs"):
        os.mkdir("rt_scraper/logs")

    log_filepath = f"rt_scraper/logs/{str(dt.datetime.now()).split('.')[0]}.log"

    logging.config.dictConfig(logging_config)
    logger = logging.getLogger("scraper_log")

    formatter = ColoredFormatter()

    file_handler = RotatingFileHandler(
        log_filepath, maxBytes=1024 * 1024 * 5, backupCount=5
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logger.setLevel("DEBUG")
    return logger
