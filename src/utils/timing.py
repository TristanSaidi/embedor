import time
import logging
from functools import wraps

def timer(func):
    """
    Decorator to measure the execution time of a function.
    Logs to 'embedor' logger with function name and arguments.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger("embedor")
        
        class_name = ""
        if args and hasattr(args[0], '__class__'):
            class_name = f"{args[0].__class__.__name__}."
        
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            
            logger.info(f"TIMER | {class_name}{func.__name__} | {elapsed:.4f}s")
            
            # Store timing (single value, not list)
            if args and hasattr(args[0], '_timings'):
                args[0]._timings[func.__name__] = (
                    args[0]._timings.get(func.__name__, 0.0) + elapsed
                )

            
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"TIMER | {class_name}{func.__name__} | FAILED after {elapsed:.4f}s | Error: {str(e)}")
            raise
    
    return wrapper