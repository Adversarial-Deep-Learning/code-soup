import logging
import sys


class Logger(object):
    def __init__(self, log_level=logging.DEBUG):
        self.logger = logging.getLogger()

        # Set global log level to 'debug' (required for handler levels to work)
        self.logger.setLevel(log_level)
        self.log_level = log_level

    def add_file_handler(self, log_file_name):
        file_handler = logging.FileHandler(log_file_name)
        file_handler.setLevel(self.log_level)
        self.logger.addHandler(file_handler)

    def add_stream_handler(self):
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(self.log_level)
        self.logger.addHandler(stream_handler)

    def debug(self, message):
        """Log debug message"""
        self.logger.debug(message)

    def info(self, message):
        """Log info message"""
        self.logger.info(message)

    def warning(self, message):
        """Log warning message"""
        self.logger.warning(message)

    def error(self, message):
        """Log error message"""
        self.logger.error(message)

    def critical(self, message):
        """Log critical message"""
        self.logger.critical(message)
