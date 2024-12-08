import time

def gen_session_id() -> str:
    """
    Get the current Unix timestamp as a string.

    Returns:
        str: The current Unix timestamp in seconds.
    """
    return str(int(time.time()))
