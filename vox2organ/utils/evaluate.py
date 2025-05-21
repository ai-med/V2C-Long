#!/usr/bin/env python3
""" Evaluation of a model """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"



import os
import sys
# remove current path from sys.path to avoid importing utils.py
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path = [p for p in sys.path if p != script_dir]
sys.path.append(os.path.join(script_dir, ".."))


import argparse
import re
import os
import glob
import random
import gc
from collections.abc import Sequence
from collections import defaultdict

import numpy as np
import torch
import trimesh
import pandas as pd
import nibabel as nib
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from pytorch3d.structures import MeshesXD
from pytorch3d.ops import iterative_closest_point


import os
import sys
import logger
from models.vox2cortex import Vox2Cortex
from utils.coordinate_transform import normalize_vertices
from utils.utils import flatten
from utils.mesh import register_mesh_to_voxel_seg
from utils.modes import TemplateModes


log = logger.get_std_logger(__name__)

# SAVE_GPU_MEMORY = False
DEVICE=None


class ModelEvaluator:
    """Class for evaluation of models.

    :param eval_dataset: The dataset split that should be used for evaluation.
    :param save_dir: The experiment directory where data can be saved.
    :param n_v_classes: Number of vertex classes.
    :param n_m_classes: Number of mesh classes.
    :param eval_metrics: A list of metrics to use for evaluation.
    :param cor_eval_metrics: A list of metrics to use for correspondence evaluation.
    :param cor_eval_icp: Whether to register the meshes before evaluating correspondence metrics.
    """

    def __init__(
        self,
        eval_dataset,
        save_dir,
        n_v_classes,
        n_m_classes,
        eval_metrics,
        cor_eval_metrics,
        cor_eval_icp=False,
        dump_graph_latent_features=False,
        template_mode=TemplateModes.STATIC,
        **kwargs,
    ):
        self._dataset = eval_dataset
        self._save_dir = save_dir
        self._n_v_classes = n_v_classes
        self._n_m_classes = n_m_classes
        self._eval_metrics = eval_metrics
        self._cor_eval_metrics = cor_eval_metrics
        self._cor_eval_icp = cor_eval_icp
        self._dump_graph_latent_features = dump_graph_latent_features
        self._template_mode = template_mode

        log.info("Created evaluator with metrics:")
        log.info(self._eval_metrics)
        log.info(self._cor_eval_metrics)



    def evaluate(
        self,
        model,
        pred_str,
        device,
        register_meshes_to_voxels=False,
        max_predictions=None,
        remove_previous_meshes=True,
        save_predictions=None,
        eval_avg_scans=False,
    ):
        with torch.no_grad():
            predictions = self.predict(
                model,
                pred_str,
                device,
                register_meshes_to_voxels,
                max_predictions,
                remove_previous_meshes,
                save_predictions,
            )
            return self.evaluate_metrics(predictions, device, eval_avg_scans)



    def predict(self,
                model,
                pred_str,
                device,
                register_meshes_to_voxels=False,
                max_predictions=None,
                remove_previous_meshes=True,
                save_predictions=None):
        """
        Generate predictions for a model.
        Returns an iterator to save memory

        :param model: The model to use for prediction.
        :param pred_str: A string that identifies the model used for
        prediction.
        :param device: The device to use for prediction.
        :param register_meshes_to_voxels: Whether to register the meshes to the
        voxel space.
        :param max_predictions: The maximum number of predictions to generate.
        :param remove_previous_meshes: Whether to remove previous meshes.
        :param save_predictions: Limit number of predictions to save.
        """
        if isinstance(model, DDP):
            model_class = model.module.__class__
        else:
            model_class = model.__class__

        # Batch size fixed for Vox2Cortex
        if model_class == Vox2Cortex:
            if isinstance(model, DDP):
                batch_size = model.module.batch_size
            else:
                batch_size = model.batch_size
        else:
            batch_size = 1

        it = range(len(self._dataset))
        if max_predictions and max_predictions < len(self._dataset):
            it = list(it)[:max_predictions]

        print("Starting predict iterations")
        cur_patient_id = None
        # Iterate over data split
        for cnt,i in enumerate(tqdm(
            it, desc=f"Predict meshes on {device}..."
        )):
            # print cuda memory usage in mbs
            # if cnt % 20 == 0:
            #     print(f"{cnt} - cuda memory allocated: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
            data = self._dataset.get_data_element(i)
            patient_id = data["patient_id"]
            template_id = data["template_id"]
            scan_id = data["scan_id"]

            cur_patient_id = patient_id

            pred_dir = self.get_pred_dir(patient_id, template_id, scan_id)

            # Affine from normalized image space to scanner RAS
            if "trans_affine_label" in data:
                vox2ras = torch.inverse(
                    torch.tensor(
                        data["trans_affine_label"], device=device
                    ).float()
                ).to(device)
            else:
                vox2ras = torch.inverse(
                    torch.tensor(
                        self._dataset.global_tarns_affine, device=device
                    ).float()
                ).to(device)

            run_model = not all(os.path.isfile(os.path.join(pred_dir, f"{mesh_name}_{pred_str}.ply"))
                for mesh_name in list(self._dataset.mesh_label_names.keys()))

            if any(isinstance(metric, Displacement) for metric in self._eval_metrics):
                run_model = True

            if run_model:
                # Generate prediction and store
                input_img = nib.Nifti1Image(
                    data["img"].cpu().numpy().astype(np.float32),
                    np.eye(4)
                )
                if logger.debug():
                    nib.save(
                        input_img,
                        os.path.join(
                            logger.get_log_dir(),
                            f"eval_input_img_{scan_id}.nii.gz"
                        )
                    )
                voxel_label = nib.Nifti1Image(
                    data["voxel_label"].cpu().numpy(),
                    np.eye(4)
                )
                if logger.debug():
                    nib.save(
                        voxel_label,
                        os.path.join(
                            logger.get_log_dir(), f"eval_label_{scan_id}.nii.gz"
                        )
                    )

                # Even though if batch_size is > 1, only one subject is
                # processed at a time
                img = data["img"].float().to(device)
                input_img = img.expand(batch_size, *img.shape)
                (template_meshes,
                template_imgs,
                template_y,
                template_points,
                template_normals,
                template_curvs,
                template_parcs) = self._dataset.get_template_meshes_batch(
                    [i]*batch_size, batch_size, device=device
                )


                # Model prediction; we need to use model.module in DDP
                # models to avoid problems in distributed training,
                # see also
                # https://discuss.pytorch.org/t/distributeddataparallel-barrier-doesnt-work-as-expected-during-evaluation/99867
                model_fwd = model.module if (
                    isinstance(model, DDP)
                ) else model

                if self._dump_graph_latent_features:
                    assert(model_class == Vox2Cortex)
                    model_fwd.graph_net.record_latent = True


                pred = model_fwd(input_img, template_meshes)

                # First mesh in the batch from the final deformation step
                mesh_pred = model_class.pred_to_final_mesh_pred(pred)

                mesh_pred = MeshesXD(
                    mesh_pred.verts_list()[: self._n_m_classes],
                    mesh_pred.faces_list()[: self._n_m_classes],
                    X_dims=(self._n_m_classes,),
                    verts_features=[fl[...,:] for fl in mesh_pred.verts_features_list()[: self._n_m_classes]],
                    virtual_edges=mesh_pred_virtual_edges.tolist() if mesh_pred._virtual_edges is not None else None,
                )


                mesh_pred = mesh_pred.transform(vox2ras)

                # we add the real total displacement from the template in the ras space as the last feature
                S,B,C = pred[0].X_dims()
                tensors_3d = pred[0].verts_features_packed()[...,-3:].reshape(S,C,-1,3)
                transformed_tensors = (vox2ras[:3,:3] @ tensors_3d.view(-1,3).T).T.view(S,C,-1,3)
                l2_norms = torch.norm(transformed_tensors, dim=-1)

                summed_l2_norms = l2_norms.sum(axis=0)
                displacements = summed_l2_norms.unsqueeze(-1)

                # dump to file
                if save_predictions is not None and cnt < save_predictions:
                    os.makedirs(pred_dir, exist_ok=True)
                    fname = os.path.join(pred_dir, f"displacements_{pred_str}.npy")
                    np.save(fname, displacements.cpu().numpy())

                assert(B==1)

                mesh_pred = MeshesXD(
                    mesh_pred.verts_list(),
                    mesh_pred.faces_list(),
                    X_dims=(self._n_m_classes,),
                    verts_features=[
                        torch.cat([
                            fl,
                            displacements[i,...]
                        ], dim=-1)
                        for i,fl in enumerate(mesh_pred.verts_features_list())],
                    virtual_edges=mesh_pred_virtual_edges.tolist() if mesh_pred._virtual_edges is not None else None,
                )

                # Undo padding etc.
                try:
                    voxel_pred = model_class.pred_to_voxel_pred(pred)[0]
                    voxel_pred = self._dataset.label_to_original_size(voxel_pred)
                except TypeError:
                    # Not all models produce a voxel pred
                    voxel_pred = None
                # Optimize mesh prediction based on voxel prediction
                if register_meshes_to_voxels:
                    mesh_pred = register_mesh_to_voxel_seg(
                        mesh_pred,
                        list(self._dataset.mesh_label_names.keys()),
                        voxel_pred,
                        list(self._dataset.voxel_label_names.keys()),
                        self._dataset.image_affine(i)
                    )

                # Save
                if save_predictions is not None and cnt < save_predictions:
                    self.save_pred(
                        patient_id,
                        template_id,
                        scan_id,
                        mesh_pred,
                        voxel_pred,
                        pred_str,
                        remove_previous_meshes,
                        model.graph_net.latent_features_log if self._dump_graph_latent_features else None,
                    )

                # explicitly free up some stuff
                del input_img, template_meshes, pred
                torch.cuda.empty_cache()
            else:
                # Load prediction
                verts, faces = [], []
                for mn in list(self._dataset.mesh_label_names.keys()):
                    mesh = trimesh.load(
                        os.path.join(pred_dir, mn + f"_{pred_str}.ply"),
                        process=False
                    )
                    verts.append(torch.tensor(mesh.vertices, device=device).float())
                    faces.append(torch.tensor(mesh.faces, device=device).long())
                mesh_pred = MeshesXD(
                    verts, faces, X_dims=(self._n_m_classes)
                ).to(device)
                try:
                    voxel_pred = nib.load(
                        os.path.join(
                            pred_dir,
                            f"pred_{pred_str}.nii.gz"
                        )
                    ).get_fdata()
                    voxel_pred = torch.tensor(voxel_pred).to(device)
                except FileNotFoundError:
                    voxel_pred = None

            yield mesh_pred, voxel_pred, data

            del mesh_pred

            del voxel_pred, data
            gc.collect()
            torch.cuda.empty_cache()










    def get_pred_dir(self, patient_id, template_id, scan_id):
        """ Get the directory where the prediction is saved """
        if self._template_mode == TemplateModes.STATIC or self._template_mode == 'STATIC':
            return os.path.join(
                self._save_dir,
                f"{scan_id}"
            )
        else:
            return os.path.join(
                self._save_dir,
                f"{patient_id}_{template_id}_{scan_id}"
            )



    def save_pred(
        self,
        patient_id,
        template_id,
        scan_id,
        mesh_pred,
        voxel_pred,
        pred_str,
        remove_previous_meshes,
        graph_latent_features=None,
    ):
        """ Save mesh and voxel prediction """
        pred_dir = self.get_pred_dir(patient_id, template_id, scan_id)
        os.makedirs(pred_dir, exist_ok=True)
        if remove_previous_meshes:
            previous_meshes = glob.glob(os.path.join(pred_dir, f"*_{pred_str}_*.ply"))
            previous_segs = glob.glob(os.path.join(pred_dir, f"*_{pred_str}_*.nii.gz"))
            for mn in previous_meshes:
                os.remove(mn)
            for seg in previous_segs:
                os.remove(seg)

        if voxel_pred is not None:
            pred_nifti = nib.Nifti1Image(
                voxel_pred.cpu().numpy(),
                self._dataset.image_affine(scan_id),
            )
            name = f"pred_{pred_str}.nii.gz"
            nii_fn = os.path.join(pred_dir, name)
            nib.save(pred_nifti, nii_fn)

        if graph_latent_features is not None:
            glf = torch.cat(graph_latent_features, dim=1)

        c = 0
        for name, v, f in zip(
            list(self._dataset.mesh_label_names.keys()),
            mesh_pred.verts_list(),
            mesh_pred.faces_list(),
        ):
            nm = f"_{pred_str}"
            trimesh.Trimesh(
                v.cpu().numpy(), f.cpu().numpy(), process=False
            ).export(os.path.join(pred_dir, name + f"{nm}.ply"))
            if graph_latent_features is not None:
                path = os.path.join(pred_dir, name + f"{nm}.npy")
                np.save(path, glf[c:c+v.shape[0]].cpu().numpy())
            c += v.shape[0]



    def evaluate_metrics(
            self,
            predictions,
            device,
            eval_avg_scans=False,
    ):
        # global DEVICE
        # DEVICE = device
        results_all = pd.DataFrame(columns=("EvalType", "PatientID", "TemplateID", "ScanID", "Tissue", "Metric", "Value"))


        # This loop is a bit complicated
        # The idea is:
        #  * we consume an item to evaluate from avg_mesh_queue
        #  * if avg_mesh_queue is empty, we consume an item from predictions
        #  * we collect the data from the current patient in patient_meshes
        #  * if we see a new patient, we evaluate the correspondences and optionally the average mesh for each patient

        cur_patient = None
        is_first_scan = True
        seen_patients = set()
        patient_meshes = defaultdict(list)

        avg_mesh_queue = []
        median_mesh_queue = []

        break_queue_empty = False

        NORMAL, AVG, MEDIAN = range(3)

        # print(self._eval_metrics)
        # print(self._cor_eval_metrics)
        while True:
            try:
                # consume next item
                if avg_mesh_queue:
                    mesh_type = AVG
                    mesh_pred, voxel_pred, data = avg_mesh_queue.pop(0)
                    scan_id, patient_id, template_id, time_step = data["scan_id"], data["patient_id"], None, data["time_step"]
                elif median_mesh_queue:
                    mesh_type = MEDIAN
                    mesh_pred, voxel_pred, data = median_mesh_queue.pop(0)
                    scan_id, patient_id, template_id, time_step = data["scan_id"], data["patient_id"], None, data["time_step"]
                else:
                    mesh_type = NORMAL
                    mesh_pred, voxel_pred, data = next(predictions)
                    scan_id, patient_id, template_id, time_step = data["scan_id"], data["patient_id"], data["template_id"], data["time_step"]


                # print(mesh_type, patient_id, scan_id, template_id, avg_mesh_queue, median_mesh_queue)

                # print(f"{scan_id=}, {patient_id=}, {template_id=}, {is_avg_mesh=}")
                # print(f"{len(avg_mesh_queue)=}")
                # print(f"{cur_patient=}")
                # print()

                # check if new patient
                is_first_scan = False
                if mesh_type == NORMAL and patient_id != cur_patient:
                    is_first_scan = True
                    # print("New patient")
                    if patient_id in seen_patients:
                        raise ValueError(f"Patient {patient_id} already seen! This means that the predictions are not sorted by patient.")

                    # we need to evluate the previous patient
                    if cur_patient is not None:
                        # if SAVE_GPU_MEMORY:
                        #     patient_meshes = {k: [(mesh_pred.to(DEVICE), voxel_pred, data) for mesh_pred, voxel_pred, data in l] for k,l in patient_meshes.items()}
                        if len(patient_meshes) > 1:
                            assert(flatten(patient_meshes.values()))
                            # evaluate correspondence metrics for all scans of patient
                            # print(f"Eval patient correspondence with {len(patient_meshes)} scans")
                            res = self.eval_correspondence(flatten(patient_meshes.values()))
                            res["EvalType"] = "correspondence"
                            res["TimeStep"] = f"{len(patient_meshes)}scans"
                            results_all = pd.concat([results_all, res], ignore_index=True)

                        if eval_avg_scans:
                            # construct and evaluate avg meshes for each scan
                            avg_meshes = []
                            median_meshes = []
                            for sid, mesh_preds_data in patient_meshes.items():
                                avg_meshes.append(self.average_meshes(mesh_preds_data))
                                median_meshes.append(self.average_meshes(mesh_preds_data, median=True))

                            avg_mesh_queue += avg_meshes
                            median_mesh_queue += median_meshes

                            if len(avg_meshes) > 1:
                                # evaluate correspondence average meshes
                                print(f"Eval avg correspondence with {len(avg_meshes)} scans")
                                res = self.eval_correspondence(avg_meshes)
                                res["EvalType"] = "avg_correspondence"
                                res["TimeStep"] = f"{len(avg_meshes)}scans"
                                results_all = pd.concat([results_all, res], ignore_index=True)

                                print(f"Eval median correspondence with {len(median_meshes)} scans")
                                res = self.eval_correspondence(median_meshes)
                                res["EvalType"] = "median_correspondence"
                                res["TimeStep"] = f"{len(median_meshes)}scans"
                                results_all = pd.concat([results_all, res], ignore_index=True)

                    cur_patient = patient_id
                    patient_meshes = defaultdict(list)
                    seen_patients.add(cur_patient)

                # print(f"{scan_id=}, {patient_id=}, {template_id=}, {time_step=}, {mesh_type=}, {is_first_scan=}")

                if len(self._eval_metrics):
                    # Load ground truth
                    if "trans_affine_label" in data:
                        # Affine from normalized image space to scanner RAS
                        vox2ras = torch.inverse(
                            torch.tensor(
                                data["trans_affine_label"], device=device
                            ).float()
                        )
                    else:
                        # log.warning("Didn't find trans affine")
                        vox2ras = torch.inverse(
                            self._dataset.global_trans_affine
                        )

                    mesh_norm_space = data["mesh_label"].to(device)
                    mesh_gt = MeshesXD(
                            mesh_norm_space.verts_list(),
                            mesh_norm_space.faces_list(),
                            X_dims=[mesh_norm_space.verts_padded().shape[0]],
                        ).to(device).transform(vox2ras)  # --> scanner RAS


                    if data["voxel_label"] is not None:
                        voxel_gt = self._dataset.label_to_original_size(
                            data["voxel_label"]
                        ).to(device)
                    else:
                        voxel_gt = None

                    if any(isinstance(metric, MeshDice) for metric in self._eval_metrics):
                        # Pred mesh in normalized image coordinates
                        dtype = mesh_pred.verts_packed().dtype
                        mesh_img_coo = mesh_pred.clone().transform(
                            torch.tensor(
                                np.linalg.inv(self._dataset.image_affine(scan_id)),
                                dtype=dtype,
                                device=device
                            )
                        )

                        _, _, affine = normalize_vertices(
                            mesh_img_coo.verts_list()[0].cpu().numpy(),  # dummy
                            voxel_gt.shape,
                            mesh_img_coo.faces_list()[0].cpu().numpy(),  # dummy
                            return_affine=True
                        )
                        mesh_pred_img_norm_coo = mesh_img_coo.transform(
                            torch.tensor(affine, dtype=dtype, device=device)
                        )
                    else:
                        mesh_pred_img_norm_coo = None

                    for metric in self._eval_metrics:
                        res = metric(
                            mesh_pred_img_norm_coo if (
                                isinstance(metric, MeshDice)
                            ) else mesh_pred,
                            mesh_gt,
                            self._n_m_classes,
                            list(self._dataset.mesh_label_names.keys()),
                            voxel_pred,
                            voxel_gt,
                            self._n_v_classes,
                            list(self._dataset.voxel_label_names.keys()),
                        )
                        res = pd.DataFrame(res)
                        res["PatientID"] = patient_id
                        res["TemplateID"] = template_id
                        res["ScanID"] = scan_id
                        res["TimeStep"] = time_step

                        if mesh_type == AVG:
                            res["EvalType"] = "meanAgg"
                        elif mesh_type == MEDIAN:
                            res["EvalType"] = "medianAgg"
                        else:
                            res["EvalType"] = "normal"
                        if results_all.empty:
                            results_all = res
                        else:
                            results_all = pd.concat([results_all, res], ignore_index=True)


                if mesh_type == NORMAL and len(self._cor_eval_metrics) > 0:
                    patient_meshes[scan_id].append((mesh_pred, voxel_pred, data))

                if break_queue_empty and len(avg_mesh_queue) == 0 and len(median_mesh_queue) == 0:
                    break
            except StopIteration:
                # if SAVE_GPU_MEMORY:
                #     patient_meshes = [(mesh_pred.to(DEVICE), voxel_pred, data) for mesh_pred, voxel_pred, data in patient_meshes.values()]
                if len(patient_meshes) > 1:
                    # evaluate correspondence metrics for all scans of patient
                    print(f"Eval correspondence with {len(flatten(patient_meshes.values()))} scans")
                    res = self.eval_correspondence(flatten(patient_meshes.values()))
                    res["EvalType"] = "correspondence"
                    res["TimeStep"] = f"{len(patient_meshes)}scans"

                    results_all = pd.concat([results_all, res], ignore_index=True)

                if eval_avg_scans:
                    break_queue_empty = True

                    # need to evaluate average meshes one last time
                    avg_meshes = []
                    median_meshes = []
                    for sid, mesh_preds_data in patient_meshes.items():
                        avg_meshes.append(self.average_meshes(mesh_preds_data))
                        median_meshes.append(self.average_meshes(mesh_preds_data, median=True))

                    avg_mesh_queue += avg_meshes
                    median_mesh_queue += median_meshes

                    if len(avg_meshes) > 1:
                        # evaluate correspondence average meshes
                        print(f"Eval avg correspondence with {len(avg_meshes)} scans")
                        res = self.eval_correspondence(avg_meshes)
                        res["EvalType"] = "avg_correspondence"
                        res["TimeStep"] = f"{len(avg_meshes)}scans"
                        results_all = pd.concat([results_all, res], ignore_index=True)

                        print(f"Eval median correspondence with {len(median_meshes)} scans")
                        res = self.eval_correspondence(median_meshes)
                        res["EvalType"] = "median_correspondence"
                        res["TimeStep"] = f"{len(avg_meshes)}scans"
                        results_all = pd.concat([results_all, res], ignore_index=True)
                else:
                    break

        return results_all




    def eval_correspondence(self,
                            mesh_preds_data: Sequence):

        meshes,_,datas = tuple(zip(*mesh_preds_data))
        # if SAVE_GPU_MEMORY:
        #     meshes = [m.to(DEVICE) for m in meshes]
        results_all = pd.DataFrame(columns=("EvalType", "PatientID", "TemplateID", "ScanID", "Tissue", "Metric", "Value"))

        if self._cor_eval_icp:
            meshes = self.icp_meshes(meshes)

        for cor_eval_metric in self._cor_eval_metrics:
            # import time
            # t0 = time.time()
            if type(cor_eval_metric) == ParcConsistency and "parc_path" in datas[-1]:
                cor_eval_metric.cache = {}
                cor_eval_metric.parc_path = datas[-1]["parc_path"]
            res = cor_eval_metric(meshes, list(self._dataset.mesh_label_names.keys()))
            res = pd.DataFrame(res)
            res["PatientID"] = datas[0]['patient_id']
            res["TemplateID"] = None
            res["ScanID"] = None
            if results_all.empty: # workaround to disable empty concat warning
                results_all = res
            else:
                results_all = pd.concat([results_all, res], ignore_index=True)
            # print(f"Eval time {cor_eval_metric}: {time.time() - t0:.2f}s")
        return results_all


    def average_meshes(self, mesh_preds_data, median=False, icp=False):
        """
        return average mesh from list of mesh predictions
        """
        meshes,voxels,datas = tuple(zip(*mesh_preds_data))

        if "trans_affine_label" in datas[0]:
            assert([np.allclose(datas[0]["trans_affine_label"], d["trans_affine_label"]) for d in datas])

        # device = meshes[0].device
        mean_mesh, median_mesh = self.average_ms(meshes, X_dims=self._n_m_classes, agg_type="median" if median else "mean", icp=icp)
        avg_mesh = median_mesh if median else mean_mesh
        if voxels[0] is None:
            avg_voxels = None
        elif median:
            avg_voxels = torch.median(torch.stack(voxels), dim=0)[0]
        else:
            avg_voxels = torch.mean(torch.stack(voxels).float(), dim=0)
        data = datas[0].copy()
        del data["template_id"]
        return avg_mesh, avg_voxels, data


    @staticmethod
    def icp_meshes(meshes):
        K = len(meshes)
        icps = iterative_closest_point(
                meshes[0].verts_packed().unsqueeze(0).expand(K-1, -1, -1),
                torch.stack([m.verts_packed() for m in meshes[1:]]),
                # verbose=True
            )
        if not icps.converged:
            log.warning("ICP did not converge for all meshes!")
        transforms = ModelEvaluator.icp_result_to_transform(icps)
        return [meshes[0]] + [m.transform(transforms[i].to(m.verts_packed().dtype)) for i,m in enumerate(meshes[1:])]

    @staticmethod
    def average_ms(meshes, X_dims, agg_type="both", icp=False):
        if icp:
            meshes = ModelEvaluator.icp_meshes(meshes)
        meshes_verts = [m.verts_list() for m in meshes]
        if agg_type in ["median", "both"]:
            median_mesh = MeshesXD(
                [torch.median(torch.stack(ms), dim=0)[0] for ms in zip(*meshes_verts)],
                meshes[0].faces_list(),
                X_dims=X_dims,
                verts_features=[torch.median(torch.stack(ms), dim=0)[0] for ms in zip(*[m.verts_features_list() for m in meshes])] if meshes[0].verts_features_list() is not None else None,
                virtual_edges=meshes[0]._virtual_edges.tolist() if meshes[0]._virtual_edges is not None else None,
            )
        else:
            median_mesh = None
        if agg_type in ["mean", "both"]:
            mean_mesh = MeshesXD(
                [torch.mean(torch.stack(ms), dim=0) for ms in zip(*meshes_verts)],
                meshes[0].faces_list(),
                X_dims=X_dims,
                verts_features=[torch.mean(torch.stack(ms), dim=0) for ms in zip(*[m.verts_features_list() for m in meshes])] if meshes[0].verts_features_list() is not None else None,
                virtual_edges=meshes[0]._virtual_edges.tolist() if meshes[0]._virtual_edges is not None else None,
            )
        else:
            mean_mesh = None
        return mean_mesh, median_mesh


    @staticmethod
    def icp_result_to_transform(icp_solution):
        """
        Convert a similarity transform to a (batch_size, 4, 4) transformation matrix.

        Args:
            icp_solution: result of iterative_closest_point

        Returns:
            A (batch_size, 4, 4) tensor representing the transformation matrices.
        """
        similarity_transform = icp_solution.RTs
        R = similarity_transform.R  # (batch_size, 3, 3)
        T = similarity_transform.T  # (batch_size, 3)
        s = similarity_transform.s  # (batch_size,)

        batch_size, _, _ = R.shape

        # Scale the rotation matrix
        R_scaled = s.view(batch_size,1,1) * R

        # Initialize a transformation matrix
        transform_matrix = torch.zeros((batch_size,4,4), device=R.device)

        # Fill the rotation and translation parts
        transform_matrix[:, :3, :3] = R_scaled
        transform_matrix[:, :3, 3] = T

        return transform_matrix
