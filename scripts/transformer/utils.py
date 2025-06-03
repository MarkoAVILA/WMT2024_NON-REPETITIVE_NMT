import ctypes
import gc
import logging
import os
import socket
import sys

from contextlib import contextmanager

import psutil

if sys.platform == "linux" or sys.platform == "linux2":
    libc = ctypes.CDLL("libc.so.6")


def init_logger(level=None):
    if level is None:
        level = "INFO"

    logging.basicConfig(
        format="%(asctime)s.%(msecs)03d000 UTC [%(module)s@%(processName)s] %(levelname)s %(message)s",
        datefmt="%Y-%b-%d %H:%M:%S",
        level=os.getenv("LOG_LEVEL", level),
    )


def get_logger():
    return logging.getLogger("pytorch_transformer")


def get_current_ram_used():
    return psutil.Process(os.getpid()).memory_info().rss / 1024**2


def free_ram():
    gc.collect()
    if sys.platform == "linux" or sys.platform == "linux2":
        libc.malloc_trim(0)


@contextmanager
def fh_out(filename=None):
    if filename:
        if isinstance(filename, str):
            fh = open(filename, "w")
        else:
            fh = filename
    else:
        fh = sys.stdout

    try:
        yield fh
    finally:
        if fh is not sys.stdout:
            fh.close()


def get_port():
    """Returns an available port number."""
    sock = socket.socket()
    sock.bind(("", 0))
    return sock.getsockname()[1]
