import time
import functools
import logging
from typing import Tuple, Type

logger = logging.getLogger(__name__)


def retry(
    retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    retry_on: Tuple[Type[Exception], ...] = (Exception,),
):
    """
    Retry decorator with exponential backoff.

    Args:
        retries: Max retry attempts
        delay: Initial delay in seconds
        backoff: Multiplier for delay
        retry_on: Exception types that trigger retry
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            current_delay = delay

            while True:
                try:
                    return func(*args, **kwargs)

                except retry_on as e:
                    attempts += 1
                    if attempts > retries:
                        logger.error(
                            f"{func.__name__} failed after {retries} retries"
                        )
                        raise

                    logger.warning(
                        f"{func.__name__} failed ({e}). "
                        f"Retry {attempts}/{retries} in {current_delay}s"
                    )

                    time.sleep(current_delay)
                    current_delay *= backoff

        return wrapper

    return decorator
