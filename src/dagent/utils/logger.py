import atexit
import logging
import logging.config
from pathlib import Path
from typing import override

# good resource: https://www.youtube.com/watch?v=9L77QExPmI0


# NOTE: the following configuration is only for applications.
# for libraries, its better to not configure so we should probably
# call setup_logging in the main application instead

DEFAULT_LOG_PATH = Path("logs/dagent.log")


class InfoFilter(logging.Filter):
    @override
    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno == logging.INFO


def setup_logging():
    DEFAULT_LOG_PATH.parent.mkdir(exist_ok=True)
    # for a small-medium sized app, one logger is sufficient. Otherwise, consider using mutiple loggers
    # for each major sub-component of the app
    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "filters": {
            "only_info": {
                "()": InfoFilter,  # instantiate InfoFilter
            }
        },
        "formatters": {
            "simple": {"format": "%(levelname)s: %(message)s"},
            "detailed": {
                "format": "[%(levelname)s|%(filename)s:%(lineno)d] %(asctime)s : %(message)s",
                # you can also do the following for alignment of log levels or use full pathname
                # "%(asctime)s [%(levelname)-8s] [%(filename)s:%(lineno)d]: %(message)s"
                "datefmt": "%Y-%m-%d %H:%M:%S%z",
            },
        },
        "handlers": {
            "stderr": {
                "class": "logging.StreamHandler",
                "formatter": "simple",
                "stream": "ext://sys.stderr",
                "level": "WARNING",
            },
            "stdout_info": {
                "class": "logging.StreamHandler",
                "formatter": "simple",
                "stream": "ext://sys.stdout",
                "filters": ["only_info"],  # attach the filter
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "DEBUG",  # lowest level to capture all logs
                "formatter": "detailed",
                "filename": DEFAULT_LOG_PATH,
                "maxBytes": 3 * 1024 * 1024,  # 3 MiB
                "backupCount": 1,  # keep up to 1 backup files
            },
            "queue_handler": {
                "class": "logging.handlers.QueueHandler",
                "handlers": ["stderr", "stdout_info", "file"],
                "respect_handler_level": True,
            },
        },
        "loggers": {"root": {"level": "DEBUG", "handlers": ["queue_handler"]}},
    }
    logging.config.dictConfig(logging_config)
    queue_handler = logging.getHandlerByName("queue_handler")
    if queue_handler is not None:
        queue_handler.listener.start()  # type: ignore[attr-defined]
        atexit.register(queue_handler.listener.stop)  # type: ignore[attr-defined]
    logging.debug(
        "\n"
        "==========================================Logging is set up.====================================================="
    )

    return logging.getLogger("dagent")


LOGGER = setup_logging()


# test main
def main():
    LOGGER.debug("This is a debug message.")
    LOGGER.info("This is an info message.")
    LOGGER.warning("This is a warning message.")
    LOGGER.error("This is an error message.")
    LOGGER.critical("This is a critical message.")
    try:
        raise ZeroDivisionError
    except ZeroDivisionError:
        LOGGER.exception("This is an exception message.")


if __name__ == "__main__":
    raise SystemExit(main())
