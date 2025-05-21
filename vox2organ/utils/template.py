
""" Utility functions for templates. """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

import os
from collections.abc import Sequence
from copy import deepcopy

import torch
import torch.nn.functional as F
import trimesh
import numpy as np
import nibabel as nib
from trimesh.scene.scene import Scene
from pytorch3d.utils import ico_sphere
from pytorch3d.structures import MeshesXD, Meshes

import logger
from utils.coordinate_transform import transform_mesh_affine


TEMPLATE_PATH = f"{os.path.dirname(__file__)}/../../supplementary_material/"

log = logger.get_std_logger(__name__)


# Specification of different templates
TEMPLATE_SPECS = {
    "fsaverage-smooth-no-parc": {
        "path": os.path.join(TEMPLATE_PATH, "brain_template", "fsaverage"),
        "mesh_suffix": "_smoothed.ply",
        "parc_labels": False,
        "group_structs": [["lh_white", "rh_white"], ["lh_pial", "rh_pial"]],
        "virtual_edges": [[0, 2], [1, 3]],
    },
    "fsaverage-no-parc": {
        "path": os.path.join(TEMPLATE_PATH, "brain_template", "fsaverage"),
        "mesh_suffix": ".ply",
        "parc_labels": False,
        "group_structs": [["lh_white", "rh_white"], ["lh_pial", "rh_pial"]],
        "virtual_edges": [[0, 2], [1, 3]],
    },
    "fsaverage-smooth-rh": {
        "path": os.path.join(TEMPLATE_PATH, "brain_template", "fsaverage"),
        "mesh_suffix": "_smoothed.ply",
        "parc_labels": False,
        "group_structs": [["rh_white"], ["rh_pial"]],
        "virtual_edges": [[0, 1]],
    },
}

def load_mesh_template(
    path: str,
    mesh_label_names: Sequence,
    group_structs,
    mesh_suffix: str=".ply",
    trans_affine=torch.eye(4),
    parc_labels=False,
    parc_feature_suffix: str=".annot",
    npy_features=False,
    npy_features_suffix=".npy",
) -> MeshesXD:

    vertices = []
    faces = []
    normals = []
    features = []

    n_groups = len(group_structs)

    # Load meshes and parcellation
    for mn in mesh_label_names:
        # Group id of the surface
        group_id = [
            i for i, x in enumerate(group_structs) if any(y in mn for y in x)
        ]
        assert len(group_id) == 1, "Group ID should be unique"
        group_id = torch.tensor(group_id[0])

        m = trimesh.load_mesh(
            os.path.join(path, mn + mesh_suffix),
            process=False
        )
        m.apply_transform(trans_affine)
        vertices.append(torch.from_numpy(m.vertices).float())
        faces.append(torch.from_numpy(m.faces).long())

        # Group ID as per-vertex feature
        surf_features = F.one_hot(
            group_id, num_classes=n_groups
        ).expand(m.vertices.shape[0], -1)

        # Parcellation as per-vertex feature
        if parc_labels:
            ft = torch.from_numpy(
                nib.freesurfer.io.read_annot(
                   os.path.join(path, mn + parc_feature_suffix)
                )[0].astype(np.int64)
            )
            # Combine -1 & 0 into one class
            ft[ft < 0] = 0
            surf_features = torch.cat([surf_features, ft.unsqueeze(1)], dim=1)

        if npy_features:
            ft = torch.from_numpy(
                np.load(
                    os.path.join(path, mn + npy_features_suffix)
                )
            )
            surf_features = torch.cat([surf_features, ft], dim=1)

        features.append(surf_features.float())

    return Meshes(vertices, faces, verts_features=features)
