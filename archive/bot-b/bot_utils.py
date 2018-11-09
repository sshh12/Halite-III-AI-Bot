from contextlib import contextmanager
import sys
import os

@contextmanager
def import_quietly():

    stderr = sys.stderr
    sys.stderr = open(os.devnull, 'w')

    yield None

    sys.stderr = stderr
