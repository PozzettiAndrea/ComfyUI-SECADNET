# SPDX-License-Identifier: GPL-3.0-or-later
# Originally from ComfyUI-CADabra: https://github.com/PozzettiAndrea/ComfyUI-CADabra
# Copyright (C) 2025 ComfyUI-CADabra Contributors

"""
Logging infrastructure for SECAD-Net operations.
"""

import logging
import time
import functools
from contextlib import contextmanager
from typing import Callable, Optional

logger = logging.getLogger("SECADNET")

_current_operation: Optional[str] = None
_operation_start: Optional[float] = None


def get_current_operation() -> tuple:
    if _current_operation is None:
        return (None, 0.0)
    return (_current_operation, time.time() - _operation_start)


@contextmanager
def log_operation(name: str, **context):
    global _current_operation, _operation_start
    ctx_str = ", ".join(f"{k}={v}" for k, v in context.items())
    full_name = f"{name}({ctx_str})" if ctx_str else name
    _current_operation = full_name
    _operation_start = time.time()
    logger.info(f"Starting: {full_name}")
    try:
        yield
        elapsed = time.time() - _operation_start
        logger.info(f"Completed: {full_name} in {elapsed:.2f}s")
    except Exception as e:
        elapsed = time.time() - _operation_start
        logger.error(f"Failed: {full_name} after {elapsed:.2f}s - {type(e).__name__}: {e}")
        raise
    finally:
        _current_operation = None
        _operation_start = None


def timed(operation_name: str = None):
    def decorator(func: Callable) -> Callable:
        name = operation_name or func.__name__
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with log_operation(name):
                return func(*args, **kwargs)
        return wrapper
    return decorator


def setup_logging(level: int = logging.INFO, log_file: str = None):
    logger.setLevel(level)
    if not logger.handlers:
        console = logging.StreamHandler()
        console.setLevel(level)
        console.setFormatter(logging.Formatter("[%(asctime)s] [SECADNET] %(message)s", datefmt="%H:%M:%S"))
        logger.addHandler(console)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        logger.addHandler(file_handler)

if not logger.handlers:
    setup_logging(level=logging.INFO)
