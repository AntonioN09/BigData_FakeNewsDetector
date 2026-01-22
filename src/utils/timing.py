import time
import logging


class Timer:
    def __init__(self, name: str):
        self.name = name
        self.start_time = None

    def __enter__(self):
        logging.info(f"[START] {self.name}")
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start_time
        logging.info(f"[END] {self.name} - {elapsed:.2f}s")
