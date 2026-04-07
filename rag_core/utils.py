import time
import logging
import itertools
from docling_core.types.doc import TableItem, TextItem

logger = logging.getLogger(__name__)

class Stopwatch:
    """A simple context manager for timing tasks and logging duration."""
    def __init__(self, label: str = 'Task') -> None:
        self.start = 0.0
        self.label = label
        
    def __enter__(self):
         logger.info(f'{self.label} started...')
         self.start = time.perf_counter()
         return self
         
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.perf_counter() - self.start
        if exc_type is None:
            logger.info(f'{self.label} completed in {duration:0.2f} seconds')
        else:
            logger.info(f'{self.label} failed after {duration:0.2f} seconds')
