
""" Test procedure """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

import json
import logging
import os
import re
import sys
from copy import deepcopy

import logger
import numpy as np
import pandas as pd
import torch
import wandb
from data.dataset_split_handler import dataset_split_handler
from data.longitudinal_info import LongitudinalInfo
from models.model_handler import ModelHandler
from params.default import DATASET_PARAMS, DATASET_SPLIT_PARAMS

from utils.evaluate import ModelEvaluator
from utils.graph_conv import *
from utils.model_names import *
from utils.modes import ExecModes
from utils.utils import dict_to_lower_dict, load_checkpoint, update_dict

log = logger.get_std_logger(__name__)

def _assemble_test_hps(hps, training_hps):
    """ Assemble the test params which are mostly equal to the training params
    but there exist exceptions.

    Warning: This check might fail if training/testing is not all with docker or all without docker (FIXED_SPLIT)
    """

    test_hps = deepcopy(training_hps)

    # Potentially different dataset
    if (hps['DATASET'] == training_hps['DATASET'] and
        (any(hps[k] != training_hps[k] for k in DATASET_SPLIT_PARAMS))):
        bad_params = [k for k in DATASET_SPLIT_PARAMS if str(hps[k]) != str(training_hps[k])]
        if len(bad_params) == 1 and isinstance(hps[bad_params[0]], list) and isinstance(hps[bad_params[0]][0], LongitudinalInfo):
            if all(str(hps[bad_params[0]][i]) == training_hps[bad_params[0]][i] for i in range(len(hps[bad_params[0]]))):
                bad_params = []

        # TODO: remove this
        if "FIXED_SPLIT" and "OVERFIT" in bad_params:
            bad_params.remove("FIXED_SPLIT")
            bad_params.remove("OVERFIT")

        # TODO: remove this too
        if "FIXED_SPLIT" in bad_params:
            bad_params.remove("FIXED_SPLIT")

        if len(bad_params) > 0:
            raise ValueError(
                f"Dataset params seem to have changed since training: {bad_params}"
            )
    test_dataset_params = {
        k: hps[k] for k in (DATASET_PARAMS + DATASET_SPLIT_PARAMS)
    }
    test_hps = update_dict(test_hps, test_dataset_params)

    # Other exceptions
    test_hps['DEVICE'] = hps['DEVICE'][0]
    test_hps['TEST_SPLIT'] = hps['TEST_SPLIT']
    test_hps['MESH_TEMPLATE_ID'] = hps['MESH_TEMPLATE_ID']
    test_hps['MODEL_CONFIG']['N_EULER_STEPS'] = hps['MODEL_CONFIG']['N_EULER_STEPS']
    test_hps['TEST_MODEL_EPOCH'] = hps['TEST_MODEL_EPOCH']
    test_hps['SANITY_CHECK_DATA'] = hps['SANITY_CHECK_DATA']
    test_hps['EVAL_METRICS'] = hps['EVAL_METRICS']
    test_hps['COR_EVAL_METRICS'] = hps['COR_EVAL_METRICS']
    test_hps['REGISTER_MESHES_TO_VOXELS'] = hps['REGISTER_MESHES_TO_VOXELS']
    test_hps['DATASET_LOW_PRECISION'] = hps['DATASET_LOW_PRECISION']

    test_hps['COR_EVAL_ICP'] = hps['COR_EVAL_ICP']
    test_hps['AGGREGATE_PATIENT_METRICS'] = hps['AGGREGATE_PATIENT_METRICS']
    test_hps['LIMIT_EVAL_SCANS'] = hps['LIMIT_EVAL_SCANS']
    test_hps['DUMP_GRAPH_LATENT_FEATURES'] = hps['DUMP_GRAPH_LATENT_FEATURES']
    print(f"{test_hps['DUMP_GRAPH_LATENT_FEATURES']=}")

    test_hps['TEMPLATE_MODE'] = hps['TEMPLATE_MODE']
    test_hps['MESH_TEMPLATE_ID'] = hps['MESH_TEMPLATE_ID']
    test_hps['TEMPLATE_DIR'] = hps['TEMPLATE_DIR']
    test_hps['TEMPLATE_SUFFIX'] = hps['TEMPLATE_SUFFIX']
    test_hps['TEMPLATE_NPY_FEATURES_SUFFIX'] = hps['TEMPLATE_NPY_FEATURES_SUFFIX']
    print(f"{test_hps['MESH_TEMPLATE_ID']=}")

    # TODO remove below
    test_hps['TEMPLATE_FEATURES'] = hps['TEMPLATE_FEATURES']

    print(f"{test_hps['COR_EVAL_METRICS']=}")


    # Warnings
    if hps['MESH_TEMPLATE_ID'] != training_hps['MESH_TEMPLATE_ID']:
        log.warning(
            "Using template %s, which is different to training template %s",
            hps['MESH_TEMPLATE_ID'],
            training_hps['MESH_TEMPLATE_ID']
        )

    # str -> object
    test_hps['MODEL_CONFIG']['GC'] = eval(test_hps['MODEL_CONFIG']['GC'])

    return test_hps


def test_routine(hps: dict, resume=False):
    """ A full testing routine for a trained model

    :param dict hps: Hyperparameters to use.
    :param resume: Only for compatibility with training but single test routine
    cannot be resumed.
    """
    experiment_name = hps['EXPERIMENT_NAME']

    if experiment_name is None:
        print("Please specify experiment name for testing with --exp_name.")
        sys.exit(1)
    if resume:
        log.warning(
            "Test routine cannot be resumed, ignoring parameter 'resume'."
        )

    # Assemble test params from current hps and training params
    param_file = logger.get_params_file()
    with open(param_file, 'r') as f:
        training_hps = json.load(f)
    test_hps = _assemble_test_hps(hps, training_hps)
    test_hps_lower = dict_to_lower_dict(test_hps)

    test_split = test_hps.get('TEST_SPLIT', 'test')
    device = test_hps['DEVICE']
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    elif device != 'cpu':
        log.warning("CUDA not available, using CPU.")
        hps['DEVICE'] = 'cpu'

    # Directoy where test results are written to
    prefix = "post_processed_" if test_hps['REGISTER_MESHES_TO_VOXELS'] else ""
    logger.set_eval_dir_name(
        os.path.join(
            prefix
            + test_split
            #+ "_v2clongpredmedian_"
            + "_template_"
            + test_hps['MESH_TEMPLATE_ID']
            + f"_{test_hps['DATASET']}"
            + f"_n_{test_hps['MODEL_CONFIG']['N_EULER_STEPS']}"
            + (f"_{test_hps['TEMPLATE_MODE']}-mode" if test_hps['TEMPLATE_MODE'] != 'STATIC' else "")
        )
    )
    test_dir = logger.get_eval_dir()
    print(f"{test_dir=}")

    log.info("Testing %s...", experiment_name)

    # Load test dataset
    log.info("Loading dataset %s...", test_hps['DATASET'])
    train_set, val_set, test_set = dataset_split_handler[test_hps['DATASET']](
        template_id=test_hps['MESH_TEMPLATE_ID'],
        save_dir=test_dir,
        load_only=test_split,
        check_dir=logger.get_log_dir(),
        **test_hps_lower
    )
    if test_split == 'validation':
        test_set = val_set
    if test_split == 'train':
        test_set = train_set
    log.info("%d test files.", len(test_set))

    evaluator = ModelEvaluator(
        eval_dataset=test_set,
        save_dir=test_dir,
        **test_hps_lower
    )

    # Test models
    model = ModelHandler[test_hps['ARCHITECTURE']].value(
        ndims=test_hps['NDIMS'],
        n_v_classes=test_hps['N_V_CLASSES'],
        n_m_classes=test_hps['N_M_CLASSES'],
        patch_size=test_hps['PATCH_SIZE'],
        **test_hps_lower['model_config']
    ).float()

    # Select best model by default or model of a certain epoch
    if test_hps['TEST_MODEL_EPOCH'] > 0:
        model_names = ["epoch_" + str(test_hps['TEST_MODEL_EPOCH']) + ".pt"]
    else:
        model_names = [
            fn for fn in os.listdir(logger.get_experiment_dir()) if (
                BEST_MODEL_NAME in fn
            )
        ]

    epochs_file = os.path.join(logger.get_experiment_dir(), "models_to_epochs.json")
    try:
        with open(epochs_file, 'r') as f:
            models_to_epochs = json.load(f)
    except FileNotFoundError:
        log.warning(
            "No models-to-epochs file found, don't know epochs of stored"
            " models."
        )
        models_to_epochs = {}
        for mn in model_names:
            models_to_epochs[mn] = -1 # -1 = unknown

    epochs_tested = []

    for mn in model_names:
        model_path = os.path.join(logger.get_experiment_dir(), mn)
        epoch = models_to_epochs.get(mn, int(test_hps['TEST_MODEL_EPOCH']))

        # Test each epoch that has been stored
        if epoch not in epochs_tested or epoch == -1:
            if epoch == -1:
                pred_str = test_hps['EXPERIMENT_NAME']
            else:
                pred_str = f'epoch_{epoch}'
            log.info(
                "Test model %s (%s) on dataset split '%s'",
                model_path, pred_str, test_split
            )

            # Avoid problem of cuda out of memory by first loading to cpu, see
            # https://discuss.pytorch.org/t/cuda-error-out-of-memory-when-load-models/38011/3
            model, _, _, _, _, _ = load_checkpoint(model, model_path, 'cpu')
            model.to(device)
            model.eval()

            results = evaluator.evaluate(
               model, pred_str, device, save_predictions=len(test_set),
               remove_previous_meshes=False,
               register_meshes_to_voxels=test_hps['REGISTER_MESHES_TO_VOXELS'],
               max_predictions=test_hps['LIMIT_EVAL_SCANS'],
               eval_avg_scans=test_hps['TEMPLATE_MODE'] in ["NXN", "NXN_SORTED"],
            )

            results.to_csv(
                os.path.join(test_dir, f"eval_results_{pred_str}.csv"),
                index=False
            )

            meaned_results = logger.df_to_wandb_log(results)
            json.dump(
                meaned_results,
                open(os.path.join(test_dir, f"eval_results_{pred_str}.json"), "w"),
                indent=4
            )

            if test_hps['AGGREGATE_PATIENT_METRICS']:
                results = results.groupby(['PatientID', 'Metric', 'Tissue', 'EvalType'])['Value'].mean().reset_index()

            results_summary = results.groupby(
                ['EvalType', 'Metric', 'Tissue']
            )['Value'].mean().reset_index()
            results_summary.to_csv(
                os.path.join(
                    test_dir,
                    f"eval_results_summary_{pred_str}.csv"
                ),
                index=False
            )
            log.info("Summary of evaluation:")
            log.info(results_summary.groupby(['EvalType', 'Metric'])['Value'].mean())
            log.info("For detailed output see " + test_dir)

            try:
                logger.wandb_test_summary(
                    test_split,
                    results,
                    hps['ENTITY'],
                    hps['PROJ_NAME'],
                    hps['EXPERIMENT_NAME']
                )
            except:
                log.debug("Writing test results to wandb not possible")

            epochs_tested.append(epoch)

    return experiment_name
