from contextlib import contextmanager
from .config import GlobalConfig
@contextmanager
def no_grad():
    previous_state = GlobalConfig.backward_mode
    GlobalConfig.backward_mode = True

    try:
        yield
    finally:
        GlobalConfig.backward_mode = previous_state