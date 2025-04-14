import logging
from .globals import Args


class Logger:
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Logger, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not Logger._initialized:
            self.setup_logger()
            Logger._initialized = True

    def setup_logger(self):
        self.logger = logging.getLogger("AppLogger")

        if not Args.ENABLE_LOGGING:
            self.logger.disabled = True
            return

        level = getattr(logging, Args.LOG_LEVEL.upper())
        self.logger.setLevel(level)

        formatter = logging.Formatter(Args.LOG_FORMAT)

        if Args.LOG_TO_FILE:

            file_handler = logging.FileHandler(Args.LOG_FILENAME)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        else:

            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

    def debug(self, message):
        self.logger.debug(message)

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def critical(self, message):
        self.logger.critical(message)


logger = Logger()
