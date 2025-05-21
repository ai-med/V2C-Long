
""" Base class for voxel2mesh models """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

from abc import ABC, abstractmethod

import logger
log = logger.get_std_logger(__name__)
import torch
import torch.nn as nn

class V2MModel(nn.Module, ABC):
    """ Base class for Voxel2Mesh models """

    def __init__(self):
        super().__init__()

        # Store state dict keys of pre-trained parameters
        self.pre_trained_ids = []

    def train(self, freeze_pre_trained=False):
        """ Overwrite train method to freeze pre-trained parameters if
        necessary.
        """
        super().train()

        if freeze_pre_trained:
            self.freeze_pre_trained()


    def freeze_pre_trained(self):
        """ Freeze pre-trained parameters.
        """
        for k, v in self.named_parameters():
            if k in self.pre_trained_ids:
                v.requires_grad = False


    def load_part(self, path: str):
        """ Load previously trained model. The loaded model can have a less
        parameters, e.g., when having it partly pre-trained.
        """
        params_all = torch.load(path)
        if 'state_dict' in params_all:
            assert 'state_dict' not in self.state_dict(),\
                    "Cannot distinguish checkpoint and model state dict."
            params_all = params_all['state_dict']
        params_self = self.state_dict()

        # Load the parts that also exist in the current model
        for k, v in params_all.items():
            if k not in self.state_dict():
                continue
            if isinstance(v, nn.Parameter):
                # Compatibility with serialized parameters
                v = v.data
            if v.shape != params_self[k].shape:
                log.warning(f"Shape mismatch when loading pre-trained model for {k}: {v.shape} vs {params_self[k].shape}. Trying to copy the parameters partially.")
                params_self[k][:, :v.shape[1]].copy_(v)
            else:
                params_self[k].copy_(v)
            # params_self[k].requires_grad = False
            self.pre_trained_ids.append(k)


    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def save(self, path):
        pass

    @staticmethod
    @abstractmethod
    def pred_to_voxel_pred(pred):
        """ Get the voxel prediction with argmax over classes applied """
        pass

    @staticmethod
    @abstractmethod
    def pred_to_raw_voxel_pred(pred):
        """ Get the voxel prediction per class """
        pass

    @staticmethod
    @abstractmethod
    def pred_to_pred_meshes(pred):
        """ Get the pytorch3d mesh predictions """
        pass

    @staticmethod
    @abstractmethod
    def pred_to_final_mesh_pred(pred):
        """ Get the final mesh prediction """
        pass

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
