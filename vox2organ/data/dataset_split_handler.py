
""" Convenient dataset splitting. """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

from data.cortex import CortexDataset
from data.supported_datasets import (
    CortexDatasets,
)

# Mapping supported datasets to split functions
dataset_split_handler = {
    **{x.name: CortexDataset.split for x in CortexDatasets},
}

