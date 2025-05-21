
""" Modes """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

from enum import IntEnum,auto

class ExecModes(IntEnum):
    """ Modes for execution """
    TRAIN = auto()
    TEST = auto()
    TRAIN_TEST = auto()
    TUNE = auto()

class DataModes(IntEnum):
    """ Modes for data """
    TRAIN = auto()
    VALIDATION = auto()
    TEST = auto()

class TemplateModes(IntEnum):
    """ Modes for data """
    STATIC = auto()
    PREV = auto()
    FIRST = auto()
    NXN = auto()
    NXN_SORTED = auto()
    MEAN = auto()
    MEDIAN = auto()
