import numpy as np
import pandas as pd
from io import StringIO


def average(arr, window):
    """Average an array arr over a certain window size
    """
    if window == 1:
        return arr
    elif window > len(arr):
        raise IndexError("Window is larger than array size")
    else:
        remainder = len(arr) % window
        if remainder == 0:
            avg = np.mean(arr.reshape(-1, window), axis=1)
        else:
            avg = np.mean(arr[:-remainder].reshape(-1, window), axis=1)
    return avg


class Log:
    """Class for analyzing log files.
    Parameters
    ----------------------
    :param filename: path to lammps log file
    :type filename: string or file
    """
    def __init__(self, filename):
        # Identifiers for places in the log file
        if hasattr(filename, "read"):
            logfile = filename
        else:
            logfile = open(filename, 'r')
        self.read_file_to_dataframe(logfile)

    def read_file_to_dataframe(self, logfile):
        # read three first lines, which should be information lines
        string = logfile.readline()[1:]  # string should start with the kws
        self.keywords = string.split()
        contents = logfile.read()
        self.contents = pd.read_table(StringIO(string + contents), sep=r'\s+')

    def find(self, entry_name):
        return np.asarray(self.contents[entry_name])

    def get_keywords(self):
        """Return list of available data columns in the log file."""
        print(", ".join(self.keywords))
