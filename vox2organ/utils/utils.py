""" Utility functions """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

import os
import math
import warnings
import collections.abc
import sys
import subprocess
from copy import deepcopy
from enum import Enum

import cupy
import linecache
import tracemalloc
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from trimesh import Trimesh
from skimage import measure
from skimage.draw import polygon
from torch.utils.dlpack import to_dlpack, from_dlpack

import logger
from utils.mesh import Mesh
from utils.coordinate_transform import (
    normalize_vertices,
    unnormalize_vertices
)

log = logger.get_std_logger(__name__)

class ExtendedEnum(Enum):
    """
    Extends an enum such that it can be converted to dict.
    """

    @classmethod
    def dict(cls):
        return {c.name: c.value for c in cls}


def create_mesh_from_file(filename: str, output_dir: str=None, store=True,
                          mc_step_size=1):
    """
    Create a mesh from file using marching cubes.

    :param str filename: The name of the input file.
    :param str output_dir: The name of the output directory.
    :param bool store (optional): Store the created mesh.
    :param int mc_step_size: The step size for marching cubes algorithm.

    :return the created mesh
    """

    name = os.path.basename(filename) # filename without path
    name = name.split(".")[0]

    data = nib.load(filename)

    img3D = data.get_fdata() # get np.ndarray
    assert img3D.ndim == 3, "Image dimension not equal to 3."

    # Use marching cubes to obtain surface mesh
    mesh = create_mesh_from_voxels(img3D, mc_step_size)

    # Store
    outfile = os.path.join(output_dir, name + ".ply") # output .ply file
    mesh = mesh.to_trimesh()

    if (output_dir is not None and store):
        mesh.export(outfile)

    return mesh

def create_mesh_from_voxels(volume, mc_step_size=1):
    """ Convert a voxel volume to mesh using marching cubes

    :param volume: The voxel volume.
    :param mc_step_size: The step size for the marching cubes algorithm.
    :return: The generated mesh.
    """
    if isinstance(volume, torch.Tensor):
        volume = volume.cpu().data.numpy()

    shape = volume.shape

    vertices_mc, faces_mc, normals, values = measure.marching_cubes(
                                    volume,
                                    0,
                                    step_size=mc_step_size,
                                    allow_degenerate=False)

    # measure.marching_cubes uses left-hand rule for normal directions, our
    # convention is right-hand rule
    faces_mc = torch.from_numpy(faces_mc).long().flip(dims=[1])

    vertices_mc, faces_mc = normalize_vertices(
        torch.from_numpy(vertices_mc).float(), shape, faces_mc
    )

    # ! Normals are not valid anymore after normalization of vertices
    normals = None

    return Mesh(vertices_mc, faces_mc, normals, values)

def create_mesh_from_pixels(img):
    """ Convert an image to a 2D mesh (= a graph) using marching squares.

    :param img: The pixel input from which contours should be extracted.
    :return: The generated mesh.
    """
    if isinstance(img, torch.Tensor):
        img = img.cpu().data.numpy()

    shape = img.shape

    vertices_ms = measure.find_contours(img)
    # Only consider main contour
    vertices_ms = sorted(vertices_ms, key=lambda x: len(x))[-1]
    # Edges = faces in 2D
    faces_ms = []
    faces_ms = torch.tensor(
        faces_ms + [[i,i+1] for i in range(len(vertices_ms) - 1)]
    )

    vertices_ms, faces_ms = normalize_vertices(
        torch.from_numpy(vertices_ms).float(), shape, faces_ms
    )

    return Mesh(vertices_ms, faces_ms)

def update_dict(d, u):
    """
    Recursive function for dictionary updating.

    :param d: The old dict.
    :param u: The dict that should be used for the update.

    :returns: A new pdated dict.
    """
    d_new = deepcopy(d)

    for k, v_u in u.items():
        if k not in d_new.keys():
            log.warn(f"Key {k} not in dict that is updated.")

        if isinstance(v_u, collections.abc.Mapping):
            v_d = d_new.get(k, {})
            v_d = v_d if isinstance(v_d, collections.abc.Mapping) else {}
            d_new[k] = update_dict(v_d, v_u)
        else:
            d_new[k] = v_u

    return d_new

def crop_slices(shape1, shape2):
    """ From https://github.com/cvlab-epfl/voxel2mesh """
    slices = [slice((sh1 - sh2) // 2, (sh1 - sh2) // 2 + sh2) for sh1, sh2 in zip(shape1, shape2)]
    return slices

def crop_and_merge(tensor1, tensor2):
    """ Crops tensor1 such that it fits the shape of tensor2 and concatenates
    both along channel dimension.
    From https://github.com/cvlab-epfl/voxel2mesh """

    slices = crop_slices(tensor1.size(), tensor2.size())
    slices[0] = slice(None)
    slices[1] = slice(None)
    slices = tuple(slices)

    return torch.cat((tensor1[slices], tensor2), 1)

def sample_outer_surface_in_voxel(volume):
    """ Samples an outer surface in 3D given a volume representation of the
    objects. This is used in wickramasinghe 2020 as ground truth for mesh
    vertices.
    """
    if volume.ndim == 3:
        a = F.max_pool3d(volume[None,None].float(), kernel_size=(3,1,1), stride=1, padding=(1, 0, 0))
        b = F.max_pool3d(volume[None,None].float(), kernel_size=(1,3,1), stride=1, padding=(0, 1, 0))
        c = F.max_pool3d(volume[None,None].float(), kernel_size=(1,1,3), stride=1, padding=(0, 0, 1))
    elif volume.ndim == 4:
        a = F.max_pool3d(volume.unsqueeze(1).float(), kernel_size=(3,1,1), stride=1, padding=(1, 0, 0))
        b = F.max_pool3d(volume.unsqueeze(1).float(), kernel_size=(1,3,1), stride=1, padding=(0, 1, 0))
        c = F.max_pool3d(volume.unsqueeze(1).float(), kernel_size=(1,1,3), stride=1, padding=(0, 0, 1))
    else:
        raise NotImplementedError
    border, _ = torch.max(torch.cat([a,b,c], dim=1), dim=1)
    if volume.ndim == 3: # back to original shape
        border = border.squeeze()
    surface = border - volume.float()
    return surface.long()


def sample_inner_volume_in_voxel(volume):
    """ Samples an inner volume in 3D given a volume representation of the
    objects. This can be seen as 'stripping off' one layer of pixels.

    Attention: 'sample_inner_volume_in_voxel' and
    'sample_outer_surface_in_voxel' are not inverse to each other since
    several volumes can lead to the same inner volume.
    """
    neg_volume = -1 * volume # max --> min
    neg_volume_a = F.pad(neg_volume, (0,0,0,0,1,1)) # Zero-pad
    a = F.max_pool3d(neg_volume_a[None,None].float(), kernel_size=(3,1,1), stride=1)[0]
    neg_volume_b = F.pad(neg_volume, (0,0,1,1,0,0)) # Zero-pad
    b = F.max_pool3d(neg_volume_b[None,None].float(), kernel_size=(1,3,1), stride=1)[0]
    neg_volume_c = F.pad(neg_volume, (1,1,0,0,0,0)) # Zero-pad
    c = F.max_pool3d(neg_volume_c[None,None].float(), kernel_size=(1,1,3), stride=1)[0]
    border, _ = torch.max(torch.cat([a,b,c], dim=0), dim=0)
    border = -1 * border
    inner_volume = torch.logical_and(volume, border)
    # Seems to lead to problems if volume.dtype == torch.uint8
    return inner_volume.type(volume.dtype)

def normalize_max_one(data):
    """ Normalize the input such that the maximum value is 1. """
    max_value = float(data.max())
    return data / max_value

def normalize_plus_minus_one(data):
    """ Normalize the input such that the values are in [-1,1]. """
    max_value = float(data.max())
    assert data.min() >= 0 and max_value > 0, "Elements should be ge 0."
    return 2 * ((data / max_value) - 0.5)

def normalize_min_max(data):
    """ Min- max normalization into [0,1] """
    min_value = float(data.min())
    return (data - min_value) / (data.max() - min_value)


def Euclidean_weights(vertices, edges):
    """ Weights for all edges in terms of Euclidean length between vertices.
    """
    weights = torch.sqrt(torch.sum(
        (vertices[edges[:,0]] - vertices[edges[:,1]])**2,
        dim=1
    ))
    return weights

def score_is_better(old_value, new_value, name):
    """ Decide whether new_value is better than old_value based on the name of
    the score.
    """
    max_scores = ('JaccardVoxel', 'JaccardMesh')
    min_scores = ('Chamfer', 'ASSD')
    if old_value is None:
        if any(name.endswith(score) for score in max_scores):
            return True, 'max'
        if any(name.endswith(score) for score in min_scores):
            return True, 'min'
        else:
            raise ValueError("Unknown score name.")

    if any(name.endswith(score) for score in max_scores):
        return new_value > old_value, 'max'
    if any(name.endswith(score) for score in min_scores):
        return new_value < old_value, 'min'
    else:
        raise ValueError("Unknown score name.")

def mirror_mesh_at_plane(mesh, plane_normal, plane_point):
    """ Mirror a mesh at a plane and return the mirrored mesh.
    The normal should point in the direction of the 'empty' side of the plane,
    i.e. the side where the mesh should be mirrored to.
    """
    # Normalize plane normal
    if not np.isclose(np.sqrt(np.sum(plane_normal ** 2)), 1):
        plane_normal = plane_normal / np.sqrt(np.sum(plane_normal ** 2))

    d = np.dot(plane_normal, plane_point)
    d_verts = -1 * (plane_normal @ mesh.vertices.T - d)
    mirrored_verts = mesh.vertices + 2 * (plane_normal[:,None] * d_verts).T

    # Flip faces to perserve normal convention
    mirrored_faces = np.flip(mesh.faces, axis=1)

    # Preserve data type
    mirrored_mesh = Trimesh(mirrored_verts, mirrored_faces)\
            if isinstance(mesh, Trimesh)\
            else Mesh(mirrored_verts, mirrored_faces)

    return mirrored_mesh

def voxelize_mesh(vertices, faces, shape, strip=True, unnormalize=True):
    """ Voxelize the mesh and return a segmentation map of 'shape' for each
    mesh class.

    :param vertices: The vertices of the mesh in packed representation
    :param faces: Corresponding faces as indices to vertices in packed
    representation
    :param shape: The shape the output image should have
    :param strip: Whether to strip the outer layer of the voxelized mesh. This
    is often a more accurate representation of the discrete volume occupied by
    the mesh.
    """
    assert len(shape) == 3, "Shape should be 3D"

    voxelized = torch.zeros(shape, dtype=torch.long)
    if unnormalize:
        unnorm_verts, faces = unnormalize_vertices(vertices, shape, faces)
    else:
        unnorm_verts, faces = vertices, faces
    pv = Mesh(unnorm_verts, faces).get_occupied_voxels(shape)
    if pv is not None:
        # Occupied voxels belong to one class
        voxelized[pv[:,0], pv[:,1], pv[:,2]] = 1
    else:
        # No mesh in the valid range predicted --> keep zeros
        pass

    # Strip outer layer of voxelized mesh
    if strip:
        voxelized = sample_inner_volume_in_voxel(voxelized)

    return voxelized

def dict_to_lower_dict(d_in: dict):
    """ Convert all keys in the dict to lower case. """
    d_out = dict((k.lower(), v) for k, v in d_in.items())
    for k, v in d_out.items():
        if isinstance(v, dict):
            d_out[k] = dict_to_lower_dict(v)

    return d_out

def voxelize_contour(vertices, shape):
    """ Voxelize the contour and return a segmentation map of shape 'shape'.
    See also
    https://stackoverflow.com/questions/39642680/create-mask-from-skimage-contour

    :param vertices: The vertices of the contour.
    :param shape: The target shape of the voxel map.
    """
    assert vertices.ndim == 3, "Vertices should be padded."
    assert vertices.shape[2] == 2 and len(shape) == 2,\
            "Method is dedicated to 2D data."
    v_shape = vertices.shape
    unnorm_verts = unnormalize_vertices(
        vertices.view(-1, 2), shape
    ).view(v_shape)
    if isinstance(unnorm_verts, torch.Tensor):
        unnorm_verts = unnorm_verts.cpu().numpy()
    voxelized_contour = np.zeros(shape, dtype=np.long)
    for vs in unnorm_verts:
        # Round to voxel coordinates
        rr, cc = polygon(vs[:,0], vs[:,1], shape)
        voxelized_contour[rr, cc] = 1

    return torch.from_numpy(voxelized_contour).long()

def edge_lengths_in_contours(vertices, edges):
    """ Compute edge lengths for all edges in 'edges'."""
    if vertices.ndim != 2 or edges.ndim != 2:
        raise ValueError("Vertices and edges should be packed.")

    vertices_edges = vertices[edges]
    v1, v2 = vertices_edges[:,0], vertices_edges[:,1]

    return torch.norm(v1 - v2, dim=1)

def choose_n_random_points(points: torch.Tensor, n: int, return_idx=False,
                           ignore_padded=False):
    """ Choose n points randomly from points. """
    if points.ndim == 3:
        res = []
        idx = []
        for k, ps in enumerate(points):
            if return_idx:
                p, i = choose_n_random_points(ps, n, return_idx, ignore_padded)
                res.append(p)
                idx += [torch.tensor([k,ii]) for ii in i]
            else:
                res.append(
                    choose_n_random_points(ps, n, return_idx, ignore_padded)
                )
        if return_idx:
            return torch.stack(res), torch.stack(idx)
        return torch.stack(res)
    if points.ndim == 2:
        n_points = len(points)
        if ignore_padded:
            n_padded = 0
            # Zero-padded points have coordinates (-1, -1, -1) after
            # normalization (in mesh coordinates)
            while (
                (points[-n_padded-1] == - torch.ones(points.shape[1])).all()
                or (points[-n_padded-1] == torch.zeros(points.shape[1])).all()
            ): n_padded += 1
            n_points = n_points - n_padded
        perm = torch.randperm(n_points)
        perm = perm[:n].sort()[0]
        if return_idx:
            return points[perm], perm
        return points[perm]

    raise ValueError("Invalid number of dimensions.")

def int_to_binlist(i, n_digits):
    """ Convert an integer to binary in the form of a list of integers with
    length n_digits, e.g.  int_to_binlist(6, 4) --> [0,1,1,0]"""
    return list(map(int, bin(i)[2:].zfill(n_digits)))


def load_checkpoint(model, model_path, device):
    checkpoint = torch.load(model_path, map_location=device)

    try:
        # new way of saving model checkpoints
        start_epoch = checkpoint['start_epoch']
        model.load_state_dict(checkpoint['state_dict'])
    except KeyError:
        # old way
        model.load_state_dict(checkpoint)
        start_epoch = -1

    return (model,
            checkpoint.get('optimizer', None),
            checkpoint.get('scheduler', None),
            checkpoint.get('best_val_score', None),
            checkpoint.get('best_val_epoch', None),
            start_epoch)


def save_checkpoint(state, save_path):

    torch.save(state, save_path)


def grad_norm(model_parameters):
    """ Track gradient norm, see
    https://discuss.pytorch.org/t/check-the-norm-of-gradients/27961
    """
    total_norm = 0.
    for p in model_parameters:
        if not p.requires_grad:
            continue
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5

    return total_norm


def global_clip_and_zscore_norm(img, cli_lower, cli_upper, mean, std):
    new_img = np.clip(img, cli_lower, cli_upper)
    new_img = (new_img - mean) / std
    return new_img


def sizeof_fmt(num, suffix="B"):
    """
    https://stackoverflow.com/a/1094933
    """
    for unit in ("", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"):
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"


def sizeof_tensor(tensor):
    return tensor.element_size() * tensor.nelement()

def sizeof_tensor_fmt(tensor):
    return sizeof_fmt(sizeof_tensor(tensor))

def flatten(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]

def call_trace(func):
    def wrapper(*args, **kwargs):
        def trace_calls(frame, event, arg):
            nonlocal depth
            if event == 'call':
                if depth == 1:
                    code = frame.f_code
                    func_name = code.co_name

                    # Extract argument values and handle torch tensors and numpy arrays
                    arg_values = []
                    for arg in frame.f_locals.values():
                        if 'torch' in str(type(arg)) and hasattr(arg, 'shape') and hasattr(arg, 'dtype'):
                            arg_values.append(f"torch.Tensor(dtype={arg.dtype}, shape={arg.shape})")
                        elif isinstance(arg, np.ndarray):
                            arg_values.append(f"numpy.ndarray(dtype={arg.dtype}, shape={arg.shape})")
                        else:
                            arg_values.append(repr(arg))
                    arg_values_str = ", ".join(arg_values)
                    print(f"calling {func_name}({arg_values_str})")

                depth += 1
            elif event == 'return':
                depth -= 1
            return trace_calls

        depth = 0
        sys.settrace(trace_calls)
        result = func(*args, **kwargs)
        sys.settrace(None)
        return result

    return wrapper


def compress_binary_tensor(tensor):
    """
    Compress a binary PyTorch tensor using CuPy's packbits.

    Args:
    - tensor (torch.Tensor): A binary tensor with values 0 or 1.

    Returns:
    - torch.Tensor: Packed tensor.
    """
    # Ensure the tensor is on the GPU and is of type Byte.
    tensor = tensor.cuda().byte()

    # Convert the tensor to a DLPack tensor.
    dlpack_tensor = to_dlpack(tensor)

    # Convert the DLPack tensor to a CuPy array.
    cupy_array = cupy.fromDlpack(dlpack_tensor)

    # Pack the bits using CuPy.
    packed_cupy_array = cupy.packbits(cupy_array)

    # Convert the packed CuPy array back to a PyTorch tensor.
    packed_tensor = from_dlpack(packed_cupy_array.toDlpack())

    return packed_tensor

def decompress_binary_tensor(packed_tensor, shape):
    """
    Decompress a packed PyTorch tensor to its original binary form.

    Args:
    - packed_tensor (torch.Tensor): The compressed tensor.
    - shape (tuple): The shape of the original tensor.

    Returns:
    - torch.Tensor: The decompressed binary tensor.
    """
    # Ensure the packed tensor is on the GPU and is of type Byte.
    packed_tensor = packed_tensor.cuda().byte()

    # Convert the packed tensor to a DLPack tensor.
    dlpack_tensor = to_dlpack(packed_tensor)

    # Convert the DLPack tensor to a CuPy array.
    cupy_array = cupy.fromDlpack(dlpack_tensor)

    # Unpack the bits using CuPy.
    unpacked_cupy_array = cupy.unpackbits(cupy_array).reshape(shape)

    # Convert the unpacked CuPy array back to a PyTorch tensor.
    tensor = from_dlpack(unpacked_cupy_array.toDlpack())

    return tensor



def get_current_rss():
    pid = os.getpid()  # Get the PID of the current process
    cmd = ["ps", "-p", str(pid), "-o", "rss="]  # Fetch RSS for the given PID
    rss = subprocess.check_output(cmd).decode('utf-8').strip()
    return int(rss)  # RSS value in kilobytes

def display_top(snapshot, key_type='lineno', limit=3):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        # replace "/path/to/module/file.py" with "module/file.py"
        filename = os.sep.join(frame.filename.split(os.sep)[-2:])
        print("#%s: %s:%s: %.1f KiB"
            % (index, filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))


import os
_proc_status = '/proc/%d/status' % os.getpid()

_scale = {'kB': 1024.0, 'mB': 1024.0*1024.0, 'KB': 1024.0, 'MB': 1024.0*1024.0}

def _VmB(VmKey):
    '''Private.'''
    global _proc_status, _scale
     # get pseudo file  /proc/<pid>/status
    try:
        t = open(_proc_status)
        v = t.read()
        t.close()
    except:
        return 0.0  # non-Linux?
     # get VmKey line e.g. 'VmRSS:  9999  kB\n ...'
    i = v.index(VmKey)
    v = v[i:].split(None, 3)  # whitespace
    if len(v) < 3:
        return 0.0  # invalid format?
     # convert Vm value to bytes
    return float(v[1]) * _scale[v[2]]

def memory(since=0.0):
    '''Return memory usage in bytes.'''
    return _VmB('VmSize:') - since

def resident(since=0.0):
    '''Return resident memory usage in bytes.'''
    return _VmB('VmRSS:') - since

def stacksize(since=0.0):
    '''Return stack size in bytes.'''
    return _VmB('VmStk:') - since
