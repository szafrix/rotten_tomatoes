import logging
import logging.config


class ColoredFormatter(logging.Formatter):
    red_bckg = "\x1b[41m"
    green_bckg = "\x1b[42m"
    reset = "\x1b[0m"
    format = "%(asctime)s - %(levelname)s - %(message)s"

    FORMATS = {
        logging.INFO: green_bckg + format + reset,
        logging.ERROR: red_bckg + format + reset,
    }

    def format(self, record: logging.LogRecord) -> str:
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


logging_config = {
    "version": 1,
    "disable_existing_loggers": True,
    "loggers": {
        "debugging_log": {
            "level": "INFO",
        }
    },
}


def get_logger() -> logging.Logger:
    logging.config.dictConfig(logging_config)
    logger = logging.getLogger("debugging_log")
    formatter = ColoredFormatter()
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)
    return logger
