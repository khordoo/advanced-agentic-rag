import time
import logging
logger = logging.getLogger(__name__)
class Stopwatch:
    def __init__(self,label='Task') -> None:
        self.start=0.0
        self.label=label
    def __enter__(self):
         logger.info(f'{self.label} started...')
         self.start=time.perf_counter()
         return self
    def __exit__(self,exc_type, exc_val, exc_tb):
        if exc_type is None:
            logger.info(f'{self.label} completed in {time.perf_counter()-self.start:0.2f} seconds')
        else:
            logger.info(f'{self.label} Failed after {time.perf_counter()-self.start:0.2f} seconds')
