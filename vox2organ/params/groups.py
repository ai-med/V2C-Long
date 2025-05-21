""" Experiment-specific parameters. """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"
import os

from params.default import hyper_ps_default
from utils.utils import update_dict
from utils.losses import *
from utils.graph_conv import (
    GraphConvNorm,
    LinearLayer,
)

_run_docker = os.path.isdir("/mnt/code")

# This dict contains groups of parameters that kind of belong together in order
# to conduct certain experiments
hyper_ps_groups = {


    ### Vox2Cortex (as presented in CVPR) ###
    'Vox2Cortex': {
        'BASE_GROUP': None,
        'BATCH_SIZE': 1,
    },

    'Vox2Cortex rh': {
        'BASE_GROUP': "Vox2Cortex",
        'STRUCTURE_TYPE': "rh-only",
        'SELECT_PATCH_SIZE': [96, 224, 192],
        'PATCH_SIZE': [96, 224, 192],
        'N_V_CLASSES': 3,
        'N_M_CLASSES': 2,
        'MESH_LOSS_FUNC_WEIGHTS': [
            [4.0] * 2,  # Chamfer
            [0.01, 0.0125],  # Cosine,
            [0.1, 0.25],  # Laplace,
            [0.001, 0.0025],  # NormalConsistency
            [5.0] * 2 # Edge
        ],
    },


    ### V2C-Flow ###

    'V2C-Flow-S': {
        'BASE_GROUP': 'Vox2Cortex',
        'BATCH_SIZE': 1,
        'MESH_LOSS_FUNC': [
            ChamferLoss(curv_weight_max=5.0),
            EdgeLoss(0),
            NormalConsistencyLoss(),
        ],
        'MESH_LOSS_FUNC_WEIGHTS': [
            [1.0] * 4,  # Chamfer
            [1.0] * 4, # Edge
            [0.0001] * 4  # NC
        ],
        'MODEL_CONFIG': {
            # S2 I5
            'GRAPH_CHANNELS': [8, 64, 64],
            'N_EULER_STEPS': 5,
            # Selected aggregation to save memory
            'AGGREGATE_INDICES': [
                [3,4,5,6,7],
                [0,1,2,8,9,10],
            ],
        },
    },

    'V2C-Flow-S rh': {
        'MESH_TEMPLATE_ID': "fsaverage-smooth-rh",
        'BASE_GROUP': 'Vox2Cortex rh',
        'BATCH_SIZE': 1,
        'MESH_LOSS_FUNC': [
            ChamferLoss(curv_weight_max=5.0),
            EdgeLoss(0),
            NormalConsistencyLoss(),
        ],
        'MESH_LOSS_FUNC_WEIGHTS': [
            [1.0] * 2,  # Chamfer
            [1.0] * 2, # Edge
            [0.0001] * 2  # NC
        ],
        'MODEL_CONFIG': {
            # S2 I5
            'GRAPH_CHANNELS': [8, 64, 64],
            'N_EULER_STEPS': 5,
            # Selected aggregation to save memory
            'AGGREGATE_INDICES': [
                [3,4,5,6,7],
                [0,1,2,8,9,10],
            ],
        },
        'DUMP_GRAPH_LATENT_FEATURES': True,
    },


    ### V2C-Long ###

    # Joint model for both hemispheres
    'V2C-Long': {
        'BASE_GROUP': 'V2C-Flow-S',
        'TEMPLATE_FEATURES': [],
        'TEMPLATE_NPY_FEATURES_SUFFIX': None,
        'TEMPLATE_MODE': 'MEAN',
        'TEMPLATE_SUFFIX': '_v2c-flow-s_base.ply',
        'TEMPLATE_DIR': '../public_experiments/v2c-flow-s_base/test_template_fsaverage-smooth-no-parc_TEST_DATASET_LONG_n_5_mean/',
        'DUMP_GRAPH_LATENT_FEATURES': False,
        'DATASET_LOW_PRECISION': True,
        'MODEL_CONFIG': {
            # Same as V2C-Flow s.t. we can use pre-trained weights
            'GRAPH_CHANNELS': [8, 64, 64],
            'ENCODER_CHANNELS': [8, 16, 32, 64, 128],
            'DECODER_CHANNELS': [64, 32, 16, 8], # Voxel decoder
            'AGGREGATE_INDICES': [
                [3,4,5,6,7],
                [0,1,2,8,9,10],
            ],
        },
    },

    # Model for right hemisphere
    'V2C-Long rh': {
        'BASE_GROUP': 'V2C-Flow-S rh',
        'TEMPLATE_FEATURES': [],
        'TEMPLATE_NPY_FEATURES_SUFFIX': '_v2c-flow-s-rh_base.npy',
        'TEMPLATE_SUFFIX': '_v2c-flow-s-rh_base.ply',
        'DUMP_GRAPH_LATENT_FEATURES': False,
        'DATASET_LOW_PRECISION': True,
        'MODEL_CONFIG': {
            'N_VERTEX_CLASSES': 2+128,
        },
        'TEMPLATE_MODE': 'MEAN',
        'TEMPLATE_DIR': '../public_experiments/v2c-flow-s-rh_base/test_template_fsaverage-smooth-rh_TEST_DATASET_LONG_n_5_mean/'
    },
}


def assemble_group_params(group_name: str):
    """ Combine group params for a certain group name and potential base
    groups.
    """
    group_params = hyper_ps_groups[group_name]
    if group_params['BASE_GROUP'] is not None:
        base_params = assemble_group_params(group_params['BASE_GROUP'])
    else:
        base_params = hyper_ps_default

    return update_dict(base_params, group_params)
