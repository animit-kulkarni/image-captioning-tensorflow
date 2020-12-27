from functools import wraps
from time import time
import logging

logger = logging.getLogger(__name__)

def timer(func):
    @wraps(func)
    def _time_it(*args, **kwargs):
        start = time()
        try:
            return func(*args, **kwargs)
        finally:
            end_ = time() - start
            logger.info(f" The function, __{func.__name__}__, took: {end_ if end_ > 0 else 0} seconds")
    return _time_it