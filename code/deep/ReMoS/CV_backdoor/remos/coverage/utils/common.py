"""
Provides some useful functions.
"""

from __future__ import absolute_import

import os
import time

import numpy as np


def readable_time_str():
    """Get readable time string based on current local time.

    The time string will be formatted as %Y-%m-%d %H:%M:%S.

    Returns
    -------
    str
        Readable time string.

    """
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())


def user_home_dir():
    """Get path of user home directory.

    Returns
    -------
    str
        Path of user home directory.

    """
    return os.path.expanduser("~")


def to_numpy(data):
    """Convert other data type to numpy. If the data itself
    is numpy type, then a copy will be made and returned.

    Returns
    -------
    numpy.array
        Numpy array of passed data.

    """
    if 'mxnet' in str(type(data)):
        data = data.asnumpy()
    elif 'torch' in str(type(data)):
        data = data.cpu().numpy()
    elif 'numpy' in str(type(data)):
        data = np.copy(data)
    return data
