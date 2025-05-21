""" Cortex dataset handler """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

import os
import random
import linecache
import warnings
import collections.abc as abc
import tracemalloc
import time
from collections import OrderedDict
from typing import Union, Sequence, Optional, Tuple, List, Dict
from abc import ABC, abstractmethod
import torch.multiprocessing as mp
NUM_PARALLEL_CALLS = min(mp.cpu_count(), 1)


import torch
import torchio as tio
import numpy as np
import nibabel as nib
import trimesh
from pytorch3d.structures import Meshes, MeshesXD
from pytorch3d.structures.utils import transform_mesh_affine
from pytorch3d.ops import sample_points_from_meshes
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

import logger
from utils.visualization import show_difference
from utils.eval_metrics import Jaccard
from utils.modes import DataModes, ExecModes, TemplateModes
from utils.mesh import Mesh, curv_from_cotcurv_laplacian
from utils.template import TEMPLATE_SPECS, load_mesh_template
from utils.utils import (
    get_current_rss,
    sizeof_fmt,
    sizeof_tensor,
    sizeof_tensor_fmt,
    compress_binary_tensor,
    decompress_binary_tensor,
    display_top,
    memory,
    resident,
    stacksize,
)


from utils.utils import voxelize_mesh, normalize_min_max, flatten
from utils.coordinate_transform import (
    normalize_vertices,
)
from data.dataset import (
    DatasetHandler,
    flip_img,
    img_with_patch_size,
)
from data.longitudinal_info import LongitudinalInfo


log = logger.get_std_logger(__name__)


def combine_labels(label, label_groups, LabelMap):
    """Combine/remap labels as specified by groups
    (see _get_seg_and_mesh_file_names). This is a bit messy but gives a lot of
    flexibility.
    """
    # Find relevant labels and set the rest to 0 (background)
    # relevant_labels = torch.tensor(
    #     [LabelMap[l_name] for group in label_groups.values() for l_name in group]
    # ).long()
    new_label = torch.zeros_like(label, dtype=label.dtype)

    # Remap label ids based on groups (ignore background here)
    for i, group in enumerate(list(label_groups.values()), start=1):
        for l_name in group:
            new_label[label == LabelMap[l_name]] = i

    return new_label

class ImageAndMeshDataset(DatasetHandler, ABC):
    """Base class for dataset handlers consisting of images, meshes, and
    segmentations.

    It loads all data specified by 'ids' directly into memory. The
    corresponding raw directory should contain a folder for each ID with
    image and mesh data.


    For initialization, it is generally recommended to use the split()
    function, which directly gives a train, validation, and test dataset.
    """

    # Generic names, are usually overridden by subclasses
    image_file_name = "mri.nii.gz"
    seg_file_name = "aseg.nii.gz"
    voxel_label_names = {"foreground": ("foreground",)}
    mesh_label_names = {"foreground": "foreground"}

    # Default value used for padding images
    PAD_VALUE = 0

    @classmethod
    @abstractmethod
    def _get_seg_and_mesh_label_names(cls, structure_type):
        """Helper function to map the structure type, i.e., a generic name
        like "white_matter", to the correct segmentation and mesh label names.

        This function should be overridden by each subclass, see for example
        the implementation in data.cortex.CortexDataset.
        """
        pass

    def __init__(
        self,
        # ids: Sequence,
        long_info: LongitudinalInfo,
        template_mode: TemplateModes,
        template_id: str,
        template_features: Sequence[str],
        template_dir: str,
        template_suffix: str,
        template_npy_features_suffix: Optional[str],
        mode: DataModes,
        raw_data_dir: str,
        patch_size,
        n_ref_points_per_structure: int,
        image_file_name: str,
        mesh_file_names: str,
        seg_file_name: Sequence[str] = None,
        voxelized_mesh_file_names: Sequence[str] = None,
        augment: bool = False,
        patch_origin: Sequence[int] = (0, 0, 0),
        select_patch_size: Sequence[int] = None,
        seg_ground_truth: str = "voxelized_meshes",
        check_dir: str = "../to_check",
        sanity_check_data: bool = True,
        dataset_low_precision: bool = False,
        device: str = "cuda:0",
        **kwargs
    ):
        mappings = long_info.get_mappings(template_mode,
                                          template_id,
                                          additional_features=template_features)

        # extract template scan pairs from mappings
        ids = list(flatten(mappings.values()))
        super().__init__(ids, mode)
        # self.ids is now set to mappings

        # dict.fromkeys is a trick to only store unique elements
        # https://stackoverflow.com/a/37163210
        self.scan_ids = list(dict.fromkeys([elem['scan'] for elem in self.ids]))
        self.template_ids = list(dict.fromkeys([elem['template'] for elem in self.ids]))

        if seg_ground_truth not in ("voxelized_meshes", "voxel_seg"):
            raise ValueError(f"Unknown seg_ground_truth {seg_ground_truth}")


        self.long_info = long_info
        self.mappings = mappings

        self.template_mode = template_mode
        self.template_id = template_id
        self.template_features = template_features
        self.template_dir = template_dir
        self.template_suffix = template_suffix
        self.template_virtual_edges = TEMPLATE_SPECS[self.template_id]["virtual_edges"]
        self.template_npy_features_suffix = template_npy_features_suffix

        self._orig_img_size = None
        self._check_dir = check_dir
        self._raw_data_dir = raw_data_dir
        self._augment = augment
        self._patch_origin = patch_origin
        self.trans_affine = OrderedDict()
        self.mesh_targets = None
        self.ndims = len(patch_size)
        self.patch_size = tuple(patch_size)
        # If not specified, select_patch_size is equal to patch_size
        self.select_patch_size = (
            select_patch_size if (select_patch_size is not None) else patch_size
        )
        self.n_m_classes = len(mesh_file_names)
        self.seg_ground_truth = seg_ground_truth
        self.n_ref_points_per_structure = n_ref_points_per_structure
        self.n_min_vertices, self.n_max_vertices = None, None
        self.n_v_classes = len(self.voxel_label_names) + 1  # +1 for background

        # Sanity checks to make sure data is transformed correctly
        self.sanity_checks = sanity_check_data

        # save memory by using lower precision to store data
        self.low_precision = dataset_low_precision

        self.device = device

        if not hasattr(self, "img_norm"):
            self.img_norm = normalize_min_max

        # Load/prepare data
        self._prepare_data_3D(
            image_file_name,
            seg_file_name,
            mesh_file_names,
            voxelized_mesh_file_names,
        )

        assert len(self.scan_ids) == len(self.voxel_labels)
        assert len(self.scan_ids) == len(self.mesh_labels)
        assert len(self.scan_ids) == len(self.trans_affine)

        assert len(self.template_ids) == len(self.templates)

        self.estimate_memory_usage()


    def estimate_memory_usage(self):
        total_memory = 0
        memory_breakdown = {"images": 0, "voxel_labels": 0, "mesh_labels": 0, "templates": 0}
        count_breakdown = {"images": 0, "voxel_labels": 0, "mesh_labels": 0, "templates": 0}

        # Estimate memory usage of images
        for img in self.images.values():
            memory = img.element_size() * img.nelement()
            memory_breakdown["images"] += memory
            count_breakdown["images"] += 1

        # Estimate memory usage of voxel_labels
        for label in self.voxel_labels.values():
            if label.is_sparse:
                # Memory of values
                values_memory = label._values().element_size() * label._nnz()
                # Memory of indices
                indices_memory = label._indices().element_size() * label._nnz() * label.dim()
                memory = values_memory + indices_memory
            else:
                memory = label.element_size() * label.nelement()

            memory_breakdown["voxel_labels"] += memory
            count_breakdown["voxel_labels"] += 1

        # Estimate memory usage of mesh_labels
        for mesh in self.mesh_labels.values():
            memory = self.estimate_mesh_memory(mesh)
            memory_breakdown["mesh_labels"] += memory
            count_breakdown["mesh_labels"] += 1

        # Estimate memory usage of templates
        for template in self.templates.values():
            memory = self.estimate_mesh_memory(template)
            memory_breakdown["templates"] += memory
            count_breakdown["templates"] += 1

        total_memory = sum(memory_breakdown.values())

        # Print total memory usage
        log.info(f'Total estimated memory usage: {total_memory / (1024 * 1024):.2f} MB')

        # Print memory usage by data type
        for data_type, memory in memory_breakdown.items():
            log.info(f'Memory used by {data_type}: {memory / (1024 * 1024):.2f} MB')

        # Print average memory usage per item
        for data_type, count in count_breakdown.items():
            average_memory = memory_breakdown[data_type] / count if count > 0 else 0
            log.info(f'Average memory used per {data_type[:-1]}: {average_memory / (1024 * 1024):.2f} MB')

    def estimate_mesh_memory(self, mesh):
        mesh_memory = 0

        # Estimate memory for vertices
        for verts in mesh.verts_list():
            mesh_memory += verts.element_size() * verts.nelement()

        # Estimate memory for faces
        for faces in mesh.faces_list():
            mesh_memory += faces.element_size() * faces.nelement()

        return mesh_memory

    def image_affine(self, scan_id):
        return nib.load(
            os.path.join(self._raw_data_dir, scan_id, self.image_file_name)
        ).affine

    @torch.no_grad()
    def _prepare_data_3D(
        self,
        image_file_name,
        seg_file_name,
        mesh_file_names,
        voxelized_mesh_file_names,
    ):
        """
        Load 3D data.

        The attributes that are set by this method are:
        - self.images
        - self.voxel_labels
        - self.voxelized_meshes
        - self.mesh_labels
        - self.trans_affine
        ^^^ these are all ordered dicts with scan_id as key
            in the order of self.scan_ids
        - self.templates
        ^^^ this is an ordered dict with template_id as key
        """

        # Image data
        self.images, img_transforms = self._load_data3D_and_transform(
            image_file_name, is_label=False, normalize=True,
        )

        log.info("Done loading mri images")

        # Voxel labels
        self.voxel_labels = None
        self.voxelized_meshes = None

        if self.sanity_checks or self.seg_ground_truth == "voxel_seg":
            self.voxel_labels, _ = self._load_data3D_and_transform(
                seg_file_name, is_label=True
            )
            self.voxel_labels = {scan_id:
                combine_labels(vl, self.voxel_label_names, self.LabelMap)
                for scan_id,vl in self.voxel_labels.items()
            }

        log.info("Done loading voxel labels")


        self.compressed_voxel_labels = False
        if self.low_precision and next(iter(self.voxel_labels.values())).max().item() == 1:
            # try to save even more memory by compressing the voxel labels
            self.voxel_label_shape = next(iter(self.voxel_labels.values())).shape

            for scan_id, label in tqdm(self.voxel_labels.items(), desc="Compressing voxel labels"):
                label_compressed = compress_binary_tensor(label)
                self.voxel_labels[scan_id] = label_compressed
            self.compressed_voxel_labels = True

        # Meshes
        self.mesh_labels = self._load_dataMesh_raw(meshnames=mesh_file_names)
        self._transform_meshes_as_images(img_transforms)

        # For now, all meshes should have the same transformation matrix
        assert all(
            np.allclose(
                self.trans_affine[self.scan_ids[0]],
                self.trans_affine[i],
            ) for i in self.scan_ids)

        log.info("Done loading mesh labels")

        self.global_trans_affine = self.trans_affine[self.scan_ids[0]]

        # templates
        self.templates = self._load_templates()

        log.info("Done loading templates")


        # Voxelize meshes if voxelized meshes have not been created so far
        # and they are required (for sanity checks or as labels)
        if self.voxelized_meshes is None and (
            self.sanity_checks or self.seg_ground_truth == "voxelized_meshes"
        ):
            self.voxelized_meshes = self._create_voxel_labels_from_meshes(
                self.mesh_labels
            )

        # Assert conformity of voxel labels and voxelized meshes
        if self.sanity_checks:
            for i, (vl, vm) in enumerate(zip(self.voxel_labels.values(), self.voxelized_meshes.values())):
                if self.compressed_voxel_labels:
                    vl = decompress_binary_tensor(vl, self.voxel_label_shape)
                iou = Jaccard(
                    vl.bool().long().cuda(),  # Combine labels
                    vm.bool().long().cuda(),  # Combine labels
                    2,
                )
                out_fn = self.scan_ids[i].replace("/", "_")
                show_difference(
                    vl.bool().long(),
                    vm.bool().long(),
                    os.path.join(
                        self._check_dir, f"diff_mesh_voxel_label_{out_fn}.png"
                    ),
                )
                if iou < 0.9:
                    log.warning(
                        f"Small IoU ({iou}) of voxel label and voxelized mesh"
                        f" label {self.scan_ids[i]}, check files at {self._check_dir}"
                    )
                    img = nib.Nifti1Image(vl.to(torch.int64).squeeze().cpu().numpy(), np.eye(4))
                    nib.save(
                        img, os.path.join(self._check_dir, "data_voxel_label.nii.gz")
                    )
                    img = nib.Nifti1Image(vm.squeeze().cpu().numpy(), np.eye(4))
                    nib.save(
                        img, os.path.join(self._check_dir, "data_mesh_label.nii.gz")
                    )
                    img = nib.Nifti1Image(
                        self.images[self.scan_ids[i]].to(torch.float32).squeeze().cpu().numpy(), np.eye(4)
                    )
                    nib.save(img, os.path.join(self._check_dir, "data_img.nii.gz"))

        # Use voxelized meshes as voxel ground truth
        if self.seg_ground_truth == "voxelized_meshes":
            self.voxel_labels = self.voxelized_meshes
            if self.compressed_voxel_labels:
                for scan_id, label in tqdm(self.voxel_labels.items(), desc="Decompressing voxel labels"):
                    label_decompressed = decompress_binary_tensor(label, self.voxel_label_shape)
                    self.voxel_labels[scan_id] = label_decompressed


    @logger.measure_time
    def get_item_from_index(self, index: int):
        """
        One data item for training.
        """
        mapping = self.ids[index]
        scan_id = mapping['scan']
        # template_id = mapping['template']

        # Raw data
        img = self.images[scan_id]
        voxel_label = self.voxel_labels[scan_id]
        if self.compressed_voxel_labels:
            voxel_label = decompress_binary_tensor(voxel_label, self.voxel_label_shape)
        mesh_label = list(self._get_mesh_target_no_faces(scan_id))
        img = img.unsqueeze(0)

        # template = self.templates[template_id]

        # Potentially augment
        if self._augment:
            # image level augmentation:
            biasfield = tio.RandomBiasField()
            gamma = tio.RandomGamma()
            noise = tio.RandomNoise(mean=0, std=(0, 0.125))

            img = gamma(img)
            img = biasfield(img)
            img = noise(img)

        log.debug("Dataset file %s", self.ids[index])

        return (index, img, voxel_label, *mesh_label)
        # return (img, voxel_label, *mesh_label)



    def get_template_meshes_batch(self, indices, batch_size, device='cuda:0'):
        """
        Get template meshes for a batch of indices.
        """
        mappings = [self.ids[i] for i in indices]
        templates = [self.templates[m['template']] for m in mappings]

        batch_template_imgs = None
        batch_template_voxel_labels = None
        batch_mesh_label = [None,None,None,None]
        additional_features = [torch.tensor([
            m[key] for key in self.template_features
        ]) for m in mappings]

        verts_list = []
        faces_list = []
        features_list = []
        C = templates[0].verts_padded().shape[0]

        for template, addf in zip(templates, additional_features):
            new_verts = template.verts_list()
            verts_list += new_verts
            faces_list += template.faces_list()
            static_features = template.verts_features_list()
            features_full = []
            for i, sf in enumerate(static_features):
                features_full.append(torch.cat([sf, addf.to(torch.float32).repeat(sf.shape[0], 1)], dim=1))
            features_list += features_full

        batch_meshes = MeshesXD(
            verts_list,
            faces_list,
            X_dims=(batch_size, C),
            verts_features=features_list,
            virtual_edges=self.template_virtual_edges
        ).to(torch.device(device))

        return (batch_meshes,
                batch_template_imgs,
                batch_template_voxel_labels,
                *batch_mesh_label)



    def get_data_element(self, index):
        """Get image, segmentation ground truth and full reference mesh. In
        contrast to 'get_item_from_index', this function is not designed to be
        wrapped by a dataloader.
        """
        mapping = self.ids[index]
        patient_id = mapping.get('patient')
        scan_id = mapping['scan']
        template_id = mapping['template']

        img = self.images[scan_id][None]
        voxel_label = self.voxel_labels[scan_id]
        if self.compressed_voxel_labels:
            voxel_label = decompress_binary_tensor(voxel_label, self.voxel_label_shape)
        mesh_label = self.mesh_labels[scan_id]
        trans_affine_label = self.trans_affine[scan_id]

        template = self.templates[template_id]

        time_step = mapping.get('scan_no')


        return {
            "img": img,
            "voxel_label": voxel_label,
            "mesh_label": mesh_label,
            "trans_affine_label": trans_affine_label,

            "mapping": mapping,
            "template": template,

            "patient_id": patient_id,
            "scan_id": scan_id,
            "template_id": template_id,
            "time_step": time_step,
        }


    def get_scan_from_index(self, scan_index):
        """
        similar to get_data_element, but index self.scan_ids and without template
        """
        scan_id = self.scan_ids[scan_index]
        img = self.images[scan_id][None]
        voxel_label = self.voxel_labels[scan_id]
        if self.compressed_voxel_labels:
            voxel_label = decompress_binary_tensor(voxel_label, self.voxel_label_shape)
        mesh_label = self.mesh_labels[scan_id]
        trans_affine_label = self.trans_affine[scan_id]

        return {
            "img": img,
            "voxel_label": voxel_label,
            "mesh_label": mesh_label,
            "trans_affine_label": trans_affine_label,
        }




    def _transform_meshes_as_images(self, img_transforms):
        """Transform meshes according to image transformations
        (crops, resize) and normalize
        """
        for i, scan_id in enumerate(tqdm(
            self.scan_ids,
            position=0,
            leave=True,
            desc="Transform meshes accordingly...",
        )):
            m,t = self.mesh_labels[scan_id], img_transforms[scan_id]
            # Transform vertices and potentially faces (to preserve normal
            # convention)
            new_vertices, new_faces = [], []
            for v, f in zip(m.verts_list(), m.faces_list()):
                new_v, new_f = transform_mesh_affine(
                    v, f, torch.tensor(t, dtype=v.dtype)
                )
                _, _, norm_affine = normalize_vertices(
                    new_v, self.patch_size, new_f, return_affine=True
                )
                new_v, new_f = transform_mesh_affine(
                    v, f, torch.tensor(norm_affine, dtype=v.dtype)
                )
                new_vertices.append(new_v)
                new_faces.append(new_f)

            # Replace mesh with transformed one
            self.mesh_labels[scan_id] = Meshes(new_vertices, new_faces)
            # Store affine transformations
            # TODO: maybe do this outside of this function
            self.trans_affine[scan_id] = norm_affine @ t @ self.trans_affine[scan_id]

    def mean_edge_length(self):
        """Average edge length in dataset.

        Code partly from pytorch3d.loss.mesh_edge_loss.
        """
        edge_lengths = []
        for m in self.mesh_labels.values():
            if self.ndims == 3:
                edges_packed = m.edges_packed()
            else:
                raise ValueError("Only 3D possible.")
            verts_packed = m.verts_packed()

            verts_edges = verts_packed[edges_packed]
            v0, v1 = verts_edges.unbind(1)
            edge_lengths.append((v0 - v1).norm(dim=1, p=2).mean().item())

        return torch.tensor(edge_lengths).mean()

    def _get_mesh_target_no_faces(self, scan_id):
        return [target[scan_id] for target in self.mesh_targets]


    def load_data3D_worker(self, args):
        fn, filename, is_label, normalize = args
        scan2img_local = {}
        scan2trans_local = {}

        img = nib.load(os.path.join(self._raw_data_dir, fn, filename))
        img_data = img.get_fdata()

        assert np.array_equal(
            np.array(img_data.shape), np.array(self._orig_img_size)
        ), "All images should be of equal size"


        img_data, trans_affine = self._get_single_patch(img_data, is_label)

        if normalize:
            img_data = self.img_norm(img_data)


        if self.low_precision:
            if is_label:
                assert(np.all(np.unique(img_data) <= 255))
                img_data = img_data.to(torch.uint8)
            else:
                img_data = img_data.to(torch.float16)

        scan2img_local[fn] = img_data
        scan2trans_local[fn] = trans_affine

        return (scan2img_local, scan2trans_local)


    def _load_data3D_and_transform(self, filename: str, is_label: bool, normalize: bool = False):
        """Load data and transform to correct patch size."""

        if self._orig_img_size is None:
            self._orig_img_size = nib.load(os.path.join(self._raw_data_dir, self.scan_ids[0], filename)).get_fdata().shape


        # Package arguments for the load_data3D_worker function
        packaged_args = [(fn, filename, is_label, normalize) for fn in self.scan_ids]

        # Run the load_data3D_worker function in parallel using process_map
        log.info(f"Loading {len(packaged_args)} images...")
        t0 = time.monotonic()
        if NUM_PARALLEL_CALLS == 1:
            results = []
            i = 0
            pbar = tqdm(total=len(packaged_args), desc="Loading images")
            while packaged_args:
                arg = packaged_args.pop()
                results.append(self.load_data3D_worker(arg))
                pbar.update(1)
                del arg
                i += 1
                if i % 100 == 0: import gc; gc.collect()
        else:
            log.info(f"Using {NUM_PARALLEL_CALLS} parallel calls to load data...")

            # Create a tqdm object
            pbar = tqdm(total=len(packaged_args), desc="Processing", dynamic_ncols=True)

            # Update function: this will be called every time a task finishes
            def update(*args):
                pbar.update()

            with mp.Pool(processes=NUM_PARALLEL_CALLS) as pool:
                # Use apply_async instead of map to allow for a callback after each task
                results = [pool.apply_async(self.load_data3D_worker, (arg,), callback=update) for arg in packaged_args]
                results = [res.get() for res in results]

            pbar.close()

        # log.info(f"Loading images took {time.monotonic() - t0:.2f} seconds ({(time.monotonic() - t0)/len(self.scan_ids):.2f} seconds per item)")

        scan2img = {}
        scan2trans = {}
        for scan2img_local, scan2trans_local in results:
            scan2img.update(scan2img_local)
            scan2trans.update(scan2trans_local)

        return scan2img, scan2trans



    def _load_templates(self):
        """
        Load template meshes
        """
        template_meshes = OrderedDict()

        template_args = TEMPLATE_SPECS[self.template_id].copy()
        template_args["trans_affine"] = self.global_trans_affine
        template_args["mesh_label_names"] = self.mesh_label_names.keys()

        del template_args["virtual_edges"]

        for template_id in tqdm(
            self.template_ids, position=0, leave=True, desc="Loading templates..."
        ):
            args = template_args.copy()
            if template_id not in TEMPLATE_SPECS:
                if self.template_dir is None:
                    raise ValueError("No template directory specified.")
                args["path"] = os.path.join(self.template_dir, template_id)
            if self.template_suffix is not None:
                args["mesh_suffix"] = self.template_suffix

            if self.template_npy_features_suffix is not None:
                args["npy_features"] = True
                args["npy_features_suffix"] = self.template_npy_features_suffix

            template_meshes[template_id] = load_mesh_template(**args)

        return template_meshes




    def _get_single_patch(self, img, is_label):
        """Extract a single patch from an image."""

        # Limits for patch selection
        lower_limit = np.array(self._patch_origin, dtype=int)
        upper_limit = np.array(self._patch_origin, dtype=int) + np.array(
            self.select_patch_size, dtype=int
        )

        # Select patch from whole image
        img_patch, trans_affine_1 = img_with_patch_size(
            img,
            self.select_patch_size,
            is_label=is_label,
            mode="crop",
            crop_at=(lower_limit + upper_limit) // 2,
            pad_value=(0 if is_label else self.PAD_VALUE),
        )
        # Zoom to certain size
        if self.patch_size != self.select_patch_size:
            img_patch, trans_affine_2 = img_with_patch_size(
                img_patch, self.patch_size, is_label=is_label, mode="interpolate"
            )
        else:
            trans_affine_2 = np.eye(self.ndims + 1)  # Identity

        trans_affine = trans_affine_2 @ trans_affine_1

        return img_patch, trans_affine

    def label_to_original_size(self, img, is_label=True):
        """Transform an image/label back to the original image size"""
        # Zoom back to original resolution
        img_zoom, _ = img_with_patch_size(
            img, self.select_patch_size, is_label=is_label, mode="interpolate"
        )
        # Invert cropping: we compute the coordinates of the original image
        # center in the coordinate frame of the cropped image
        center_cropped_cropped_frame = np.array(self.patch_size) // 2
        lower_limit = np.array(self._patch_origin, dtype=int)
        upper_limit = np.array(self._patch_origin, dtype=int) + np.array(
            self.select_patch_size, dtype=int
        )
        center_cropped_orig_frame = (lower_limit + upper_limit) // 2
        center_orig_orig_frame = np.array(self._orig_img_size) // 2
        delta_centers = center_orig_orig_frame - center_cropped_orig_frame
        center_orig_cropped_frame = center_cropped_cropped_frame + delta_centers
        img_orig, _ = img_with_patch_size(
            img_zoom.cpu().numpy(),
            self._orig_img_size,
            is_label=is_label,
            mode="crop",
            crop_at=center_orig_cropped_frame,
            pad_value=(0 if is_label else self.PAD_VALUE),
        )

        return img_orig

    def _create_voxel_labels_from_meshes(self, mesh_labels):
        """Return the voxelized meshes as 3D voxel labels."""
        data = OrderedDict()
        # Iterate over subjects
        for i, (scan_id, m) in enumerate(tqdm(
            list(mesh_labels.items()), position=0, leave=True, desc="Voxelize meshes..."
        )):
            voxelized = torch.zeros(self.patch_size, dtype=torch.long)
            # Iterate over meshes
            for j, (v, f) in enumerate(zip(m.verts_list(), m.faces_list()), start=1):
                voxel_label = voxelize_mesh(v, f, self.patch_size)
                voxelized[voxel_label != 0] = j

            data[scan_id] = voxelized

        return data


    def _load_dataMesh_raw(self, meshnames):
        """Load mesh such that it's registered to the respective 3D image. If
        a mesh cannot be found, a dummy is inserted if it is a test split.
        """
        data = OrderedDict()
        assert len(self.trans_affine) == 0, "Should be empty."
        for fn in tqdm(self.scan_ids, position=0, leave=True, desc="Loading meshes..."):
            # Voxel coords
            orig = nib.load(os.path.join(self._raw_data_dir, fn, self.image_file_name))
            vox2world_affine = orig.affine
            world2vox_affine = np.linalg.inv(vox2world_affine)
            self.trans_affine[fn] = world2vox_affine
            file_vertices = []
            file_faces = []
            for mn in meshnames:
                try:
                    mesh = trimesh.load_mesh(
                        os.path.join(self._raw_data_dir, fn, mn + ".stl")
                    )
                except ValueError:
                    try:
                        mesh = trimesh.load_mesh(
                            os.path.join(self._raw_data_dir, fn, mn + ".ply"),
                            process=False,
                        )
                    except Exception as e:
                        # Insert a dummy if dataset is test split
                        if self.mode != DataModes.TEST:
                            raise e
                        mesh = trimesh.creation.icosahedron()
                        log.warning(f"No mesh for file {fn}/{mn}," " inserting dummy.")
                # World --> voxel coordinates
                mesh.apply_transform(world2vox_affine)
                # Store min/max number of vertices
                self.n_max_vertices = (
                    np.maximum(mesh.vertices.shape[0], self.n_max_vertices)
                    if (self.n_max_vertices is not None)
                    else mesh.vertices.shape[0]
                )
                self.n_min_vertices = (
                    np.minimum(mesh.vertices.shape[0], self.n_min_vertices)
                    if (self.n_min_vertices is not None)
                    else mesh.vertices.shape[0]
                )
                # Add to structures of file
                file_vertices.append(torch.from_numpy(mesh.vertices).float())
                file_faces.append(torch.from_numpy(mesh.faces).long())

            # Treat as a batch of meshes
            mesh = Meshes(file_vertices, file_faces)
            data[fn] = mesh

        return data


    @torch.no_grad()
    def create_training_targets(self, remove_meshes=False):
        """Sample surface points, normals and curvaturs from meshes."""
        if self.mesh_labels[self.scan_ids[0]] is None:
            warnings.warn(
                "Mesh labels do not exist (anymore) and no new training"
                " targets can be created."
            )
            return self.mesh_labels

        # from IPython import embed; embed()
        if type(self.device) == list:
            d = self.device[0]
        else:
            d = self.device

        points, normals, curvs = OrderedDict(), OrderedDict(), OrderedDict()

        N = 1
        # we use N GB chunks to avoid cuda OOM
        max_mem_usage = N * (1024 ** 3)
        mesh_memory_bytes = self.estimate_mesh_memory(list(self.mesh_labels.values())[0])
        chunk_size = max_mem_usage // mesh_memory_bytes
        print(f"{mesh_memory_bytes=}, {chunk_size=}")
        print(f"Number of chunks: {len(self.scan_ids) // chunk_size + 1}")

        meshes = [self.mesh_labels[scan_id] for scan_id in self.scan_ids]
        mesh_chunks = [meshes[i:min(i+chunk_size, len(meshes))]
                       for i in range(0, len(meshes), chunk_size)]

        t0 = time.time()  # Start timing for the entire process

        n_classes = len(meshes[0].verts_list())

        for mesh_chunk in mesh_chunks:
            # Move entire chunk to GPU
            mesh_chunk_gpu = [m.to(d) for m in mesh_chunk]

            chunk_curv_list = []
            for m in mesh_chunk_gpu:
                # Compute curvature for the mesh
                curv_list_mesh = [
                    curv_from_cotcurv_laplacian(v, f).unsqueeze(-1)
                    for v, f in zip(m.verts_list(), m.faces_list())
                ]
                chunk_curv_list += curv_list_mesh

            m_chunk_new = Meshes(flatten([m.verts_list() for m in mesh_chunk_gpu]),
                                flatten([m.faces_list() for m in mesh_chunk_gpu]),
                                verts_features=chunk_curv_list)  # Use chunk-specific curvatures

            # Use sample_points_from_meshes on the chunk
            p_chunk, n_chunk, c_chunk = sample_points_from_meshes(
                m_chunk_new,
                self.n_ref_points_per_structure,
                return_normals=True,
                interpolate_features="barycentric",
            )

            # Fill up points, normals, and curvs dictionaries
            start_idx_for_chunk = len(points)
            for i in range(len(mesh_chunk)):
                scan_id = self.scan_ids[start_idx_for_chunk + i]
                points[scan_id] = p_chunk[n_classes*i:n_classes*(i+1)].to('cpu')
                normals[scan_id] = n_chunk[n_classes*i:n_classes*(i+1)].to('cpu')
                curvs[scan_id] = c_chunk[n_classes*i:n_classes*(i+1)].to('cpu')

            # cuda clear cache
            torch.cuda.empty_cache()

        elapsed_time = time.time() - t0  # Calculate elapsed time for processing all chunks
        log.info(f"Processing all chunks took {elapsed_time} seconds ({elapsed_time / len(meshes)} seconds per mesh)")

        # Placeholder for point labels
        point_classes = OrderedDict((scan_id,torch.zeros_like(curvs[scan_id]))
                                    for scan_id in self.scan_ids)

        self.mesh_targets = (points, normals, curvs, point_classes)

        return self.mesh_targets



    def augment_data(self, img, label, coordinates, normals):
        assert self._augment, "No augmentation in this dataset."
        return flip_img(img, label, coordinates, normals)

    def check_augmentation_normals(self):
        """Assert correctness of the transformation of normals during
        augmentation.
        """
        py3d_mesh = self.mesh_labels[self.scan_ids[0]]
        _, _, coo_f, normals_f = self.augment_data(
            self.images[self.scan_ids[0]].numpy(),
            self.voxel_labels[self.scan_ids[0]].numpy(), #TODO: might need to decompress
            py3d_mesh.verts_padded(),
            py3d_mesh.verts_normals_padded(),
        )
        py3d_mesh_aug = Meshes(coo_f, py3d_mesh.faces_padded())
        # Assert up to sign of direction
        assert torch.allclose(
            normals_f, py3d_mesh_aug.verts_normals_padded(), atol=7e-03
        ) or torch.allclose(
            -normals_f, py3d_mesh_aug.verts_normals_padded(), atol=7e-03
        )


    @classmethod
    def split(
        cls,
        raw_data_dir,
        template_id: str,
        template_mode: Union[str, TemplateModes],
        augment_train: bool = False,
        dataset_seed: int = 0,
        all_ids_file: str = None,
        dataset_split_proportions: Sequence[int] = None,
        fixed_split: Union[dict, Sequence[str]] = None,
        overfit: int = None,
        save_dir: str = None,
        load_only: Union[str, Sequence[str]] = ("train", "validation", "test"),
        **kwargs,
    ):
        """Create train, validation, and test split of data"

        :param str raw_data_dir: The raw base folder; should contain a folder for each
        ID
        :param augment_train: Augment training data.
        :param dataset_seed: A seed for the random splitting of the dataset.
        :param all_ids_file: A file that contains all IDs that should be taken
        into consideration.
        :param dataset_split_proportions: The proportions of the dataset
        splits, e.g. (80, 10, 10)
        :param fixed_split: A dict containing file ids for 'train',
        'validation', and 'test'. If specified, values of dataset_seed,
        overfit, and dataset_split_proportions will be ignored. Alternatively,
        a sequence of files containing ids can be given.
        :param overfit: Create small datasets for overfitting if this parameter
        is > 0.
        :param load_only: Only return the splits specified (in the order train,
        validation, test, while missing splits will be None). This is helpful
        to save RAM.
        :param kwargs: Parameters of ImageAndMeshDataset + subclass-specific
        parameters.
        :return: (Train dataset, Validation dataset, Test dataset)
        """

        if isinstance(template_mode, str):
            template_mode = TemplateModes[template_mode]

        # step 1: get LongitudinalInfo objects for each split

        # Decide between fixed and random split
        if isinstance(fixed_split, abc.Sequence) and isinstance(fixed_split[0], LongitudinalInfo):
            li_train, li_val, li_test = fixed_split
        else:
            # determine the ids to use
            if isinstance(fixed_split, dict):
                files_train = fixed_split["train"]
                files_val = fixed_split["validation"]
                files_test = fixed_split["test"]
            elif isinstance(fixed_split, abc.Sequence):
                assert len(fixed_split) == 3, "Should contain one file per split"
                convert = lambda x: x[:-1]  # 'x\n' --> 'x'
                train_split = os.path.join(raw_data_dir, fixed_split[0])
                try:
                    files_train = list(map(convert, open(train_split, "r").readlines()))
                except FileNotFoundError:
                    files_train = []
                    log.warning("No training files.")
                val_split = os.path.join(raw_data_dir, fixed_split[1])
                try:
                    files_val = list(map(convert, open(val_split, "r").readlines()))
                except FileNotFoundError:
                    files_val = []
                    log.warning("No validation files.")
                test_split = os.path.join(raw_data_dir, fixed_split[2])
                try:
                    files_test = list(map(convert, open(test_split, "r").readlines()))
                except FileNotFoundError:
                    files_test = []
                    log.warning("No test files.")
            elif fixed_split is None:
                # Random split
                assert (
                    np.sum(dataset_split_proportions) == 100
                ), "Splits need to sum to 100."
                allids = [line.rstrip()
                            for line in open(all_ids_file, 'r').readlines()]
                random.Random(dataset_seed).shuffle(allids)
                indices_train = slice(
                    0, dataset_split_proportions[0] * len(allids) // 100
                )
                indices_val = slice(
                    indices_train.stop,
                    indices_train.stop
                    + (dataset_split_proportions[1] * len(allids) // 100),
                )
                indices_test = slice(indices_val.stop, len(allids))
                files_train = allids[indices_train]
                files_val = allids[indices_val]
                files_test = allids[indices_test]

            li_train = LongitudinalInfo(files_train)
            li_val = LongitudinalInfo(files_val)
            li_test = LongitudinalInfo(files_test)


        if overfit:
            li_train.set_overfit(overfit)
            li_val = li_train
            li_test = li_train


        DatasetHandler.save_ids(li_train.get_mappings(template_mode, template_id),
                         li_val.get_mappings(template_mode, template_id),
                         li_test.get_mappings(template_mode, template_id),
                         save_dir)

        # Create train, validation, and test datasets
        if "train" in load_only:
            li_train.dump(os.path.join(save_dir, "train_split.csv"))
            train_dataset = cls(
                long_info=li_train,
                template_id=template_id,
                template_mode=template_mode,
                raw_data_dir=raw_data_dir,
                mode=DataModes.TRAIN,
                augment=augment_train,
                **kwargs,
            )
        else:
            train_dataset = None

        if 'reduced_gt' in kwargs:
            kwargs.pop('reduced_gt')

        if "validation" in load_only:
            li_val.dump(os.path.join(save_dir, "val_split.csv"))
            val_dataset = cls(
                long_info=li_val,
                template_id=template_id,
                template_mode=template_mode,
                raw_data_dir=raw_data_dir,
                mode=DataModes.VALIDATION,
                augment=False,
                reduced_gt=False,
                **kwargs,
            )
        else:
            val_dataset = None

        if "test" in load_only:
            li_test.dump(os.path.join(save_dir, "test_split.csv"))
            test_dataset = cls(
                long_info=li_test,
                template_id=template_id,
                template_mode=template_mode,
                raw_data_dir=raw_data_dir,
                mode=DataModes.TEST,
                augment=False,
                reduced_gt=False,
                **kwargs,
            )
        else:
            test_dataset = None

        return train_dataset, val_dataset, test_dataset
