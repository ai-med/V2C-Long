
""" Evaluation metrics. Those metrics are typically computed directly from the
model prediction, i.e., in normalized coordinate space unless specified
otherwise."""

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

from abc import ABC, abstractmethod

import numpy as np
import torch
import pandas as pd
import pymeshlab as pyml
from scipy.spatial.distance import dice

import logger

log = logger.get_std_logger(__name__)


def Jaccard(pred, target, n_classes):
    """ Jaccard/Intersection over Union """
    ious = []
    # Ignoring background class 0
    for c in range(1, n_classes):
        pred_idxs = pred == c
        target_idxs = target == c
        intersection = pred_idxs[target_idxs].long().sum().data.cpu()
        union = (
            pred_idxs.long().sum().data.cpu() +
            target_idxs.long().sum().data.cpu() -
            intersection
        )
        # +1 for smoothing (no division by 0)
        ious.append(float(intersection + 1) / float(union + 1))

    # Return average iou over classes ignoring background
    return np.sum(ious)/(n_classes - 1)
