
""" Put module information here """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

import os
from enum import IntEnum, auto

from data.longitudinal_info import LongitudinalInfo


# Discover whether application is running in docker
_run_docker = os.path.isdir("/mnt/code")

prefix = "/mnt/data/" if _run_docker else "/home/local/"
prefix_project = "/mnt/data/" if _run_docker else "/home/local/project/"


class SupportedDatasets(IntEnum):
    """ List supported datasets """

    TEST_DATASET_LONG = auto()


class CortexDatasets(IntEnum):
    """ List cortex datasets """

    TEST_DATASET_LONG = SupportedDatasets.TEST_DATASET_LONG.value


dataset_paths = {
    SupportedDatasets.TEST_DATASET_LONG.name: {
        'RAW_DATA_DIR' : "/path/to/longitudinal/dataset/subjects_dir/",
        'FIXED_SPLIT' : [
            LongitudinalInfo(f"../supplementary_material/example_data.csv", scan_col="IMAGEUID"),
            LongitudinalInfo(f"../supplementary_material/example_data.csv", scan_col="IMAGEUID"),
            LongitudinalInfo(f"../supplementary_material/example_data.csv", scan_col="IMAGEUID"),
        ]
    },
}
