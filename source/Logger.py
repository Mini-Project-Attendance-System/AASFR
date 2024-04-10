import logging
import os
from datetime import datetime

class Logger:

    def __init__(self, logger_name):

        self.logger_name = logger_name
        self.logger = logging.getLogger(logger_name)

        try:

            if os.path.exists("./logs"):

                os.path.isdir("./logs")

            else:

                os.makedirs("./logs", exist_ok=True)

        except Exception as e:

            return -2

    def toggle_logger(self, logger_state):

        if logger_state:

            self.logger.setLevel(logging.DEBUG)

            console_handle = logging.StreamHandler()
            file_handle = logging.FileHandler(
                f'./logs/{self.logger_name}_{datetime.now().__str__().replace("-", "").replace(":", "").replace(" ", "").replace(".", "")}.log', mode="w+")

            console_handle.setLevel(logging.DEBUG)
            file_handle.setLevel(logging.DEBUG)

            console_handle.setFormatter(logging.Formatter(
                "%(name)s > %(levelname)s > %(message)s"))
            file_handle.setFormatter(logging.Formatter(
                "%(process)d > %(asctime)s > %(name)s > %(levelname)s > %(message)s"))

            self.logger.addHandler(console_handle)
            self.logger.addHandler(file_handle)

        else:

            self.logger.setLevel(100)
