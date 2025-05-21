#!/usr/bin/env python3
"""
Script to generate mean/median meshes for usage as templates for v2c-long

./create_mean_meshes.py /path/to/v2c-long/experiments/v2c-flow-s_base/test...

It will create a new directory called {orig_dir}_mean or {orig_dir}_median
With the following structure:
{orig_dir}_mean
├── PATIENT_ID1
│   ├── lh_pial.ply
│   ├── lh_white.ply
│   ├── ...
├── PATIENT_ID2
│   ├── lh_pial.ply
│   ├── lh_white.ply
│   ├── ...
├── ...
"""
import argparse
import glob
import os
import os.path
import re
import sys
import time

from multiprocessing import Pool

import pandas as pd
import torch
import trimesh
import numpy as np

from pytorch3d.structures import MeshesXD
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.evaluate import ModelEvaluator

from tqdm.contrib.concurrent import process_map



def get_patient_groups(csv_file):
    df = pd.read_csv(csv_file)
    # create a dictionary with patient IDs as keys and a list of scan IDs as values
    # PTID, IMAGEUID are the column names
    patient_groups = df.groupby("PTID")["IMAGEUID"].apply(list).to_dict()
    return patient_groups


def process_item(suffix, input_dir, aggregate_type, icp, patient_id, scan_ids, device, v2cc, remove_source_files):
    do_patient = False
    for name in ["mean", "median"]:
        if aggregate_type not in [name, "both"]: continue
        output_dir = input_dir.rstrip("/") + "_" + name + ("_icp" if icp else "")
        patient_output_dir = os.path.join(output_dir, str(patient_id))
        do_patient = do_patient or (not os.path.exists(patient_output_dir))
    if not do_patient: return

    t0 = time.perf_counter()
    if not v2cc:
        fnames = [os.path.basename(f) for f in glob.glob(os.path.join(input_dir, str(scan_ids[0]), f"*{suffix}.ply"))]
    else:
        fnames = [os.path.basename(f) for f in glob.glob(os.path.join(input_dir, f"{scan_ids[0]}_epoch*_struc*_meshpred.ply"))] # warning: this averages over epochs too
        fnames = [f.replace(f"{scan_ids[0]}", "") for f in fnames]
    t1 = time.perf_counter()
    meshes = []
    for scan_id in scan_ids:
        if not v2cc:
            meshes_t = [trimesh.load(os.path.join(input_dir, str(scan_id), f"{fname}"), process=False) for fname in fnames]
        else:
            meshes_t = [trimesh.load(os.path.join(input_dir, f"{scan_id}{fname}"), process=False) for fname in fnames]
        assert(all([m.vertices.shape[0] == meshes_t[0].vertices.shape[0] for m in meshes_t]))
        meshes.append(MeshesXD(verts=[torch.tensor(m.vertices, device=device) for m in meshes_t], faces=[torch.tensor(m.faces, device=device) for m in meshes_t], X_dims=(len(fnames,),)).to(device))

    mean_mesh,median_mesh = ModelEvaluator.average_ms(meshes, X_dims=(len(fnames),), agg_type=aggregate_type, icp=icp)
    # mean_mesh,median_mesh = None,None


    # average features
    if not v2cc:
        fnames_npy = [os.path.basename(f) for f in glob.glob(os.path.join(input_dir, str(scan_ids[0]), f"*{suffix}.npy")) if "displacements" not in f]
    else:
        fnames_npy = [os.path.basename(f) for f in glob.glob(os.path.join(input_dir, f"{scan_ids[0]}_epoch*_struc*_meshpred.npy")) if "displacements" not in f] # warning: this averages over epochs too
        fnames_npy = [f.replace(f"{scan_ids[0]}", "") for f in fnames_npy]
    fs = []
    for scan_id in scan_ids:
        if not v2cc:
            features = [np.load(os.path.join(input_dir, str(scan_id), f"{fname}")) for fname in fnames_npy]
        else:
            features = [np.load(os.path.join(input_dir, f"{scan_id}{fname}")) for fname in fnames_npy]
        fs.append(features)

    t2 = time.perf_counter()

    fs = np.array(fs)
    if aggregate_type in ["median", "both"]:
        median_features = np.median(fs, axis=0)
    else:
        median_features = None
    if aggregate_type in ["mean", "both"]:
        mean_features = np.mean(fs, axis=0)
    else:
        mean_features = None

    t3 = time.perf_counter()

    for avg_mesh,avg_features,name in zip([mean_mesh,median_mesh], [mean_features,median_features], ["mean", "median"]):
        if avg_mesh is None: continue
        output_dir = input_dir.rstrip("/") + "_" + name + ("_icp" if icp else "")
        patient_output_dir = os.path.join(output_dir, str(patient_id))
        os.makedirs(patient_output_dir, exist_ok=True)
        vertices_l, faces_l = avg_mesh.verts_list(), avg_mesh.faces_list()
        for fname, verts, faces in zip(fnames, vertices_l, faces_l):
            assert(trimesh.Trimesh(vertices=verts.to("cpu").numpy(), faces=faces.to("cpu").numpy(), process=False).vertices.shape[0] == 163842)
            trimesh.Trimesh(vertices=verts.to("cpu").numpy(), faces=faces.to("cpu").numpy(), process=False).export(os.path.join(patient_output_dir, fname))
        for fname_npy, af in zip(fnames_npy, avg_features):
            np.save(os.path.join(patient_output_dir, fname_npy), af)

    t4 = time.perf_counter()
    # print(f"Processed {patient_id} in {t4-t0:.2f}s (load ply {t1-t0:.2f}s, load npy {t2-t1:.2f}s, avg {t3-t2:.2f}s, save {t4-t3:.2f}s)")

    if remove_source_files:
        for scan_id in scan_ids:
            if not v2cc:
                for fname in fnames:
                    os.remove(os.path.join(input_dir, str(scan_id), fname))
                for fname_npy in fnames_npy:
                    os.remove(os.path.join(input_dir, str(scan_id), fname_npy))
            else:
                for fname in fnames:
                    os.remove(os.path.join(input_dir, f"{scan_id}{fname}"))
                for fname_npy in fnames_npy:
                    os.remove(os.path.join(input_dir, f"{scan_id}{fname_npy}"))

def process_item_worker(args):
    return process_item(*args)


def main(input_dir, csv_file, icp=True, aggregate_type="mean", device="cuda:0", skip=0, max=None, v2cc=False, remove_source_files=False):
    if aggregate_type in ["mean", "both"]:
        print(f"saving meshes to {input_dir.rstrip('/')}_mean" + ("_icp" if icp else ""))
    if aggregate_type in ["median", "both"]:
        print(f"saving meshes saved to {input_dir.rstrip('/')}_median" + ("_icp" if icp else ""))

    patient_groups = get_patient_groups(csv_file)

    if skip > 0:
        print(f"Skipping {skip} patients")
        patient_groups = {k:v for i,(k,v) in enumerate(patient_groups.items()) if i >= skip}
    if max is not None:
        print(f"Limiting to {max} patients")
        patient_groups = {k:v for i,(k,v) in enumerate(patient_groups.items()) if i < max}

    example_scan = next(iter(patient_groups.values()))[0]
    if not v2cc:
        example_dir = os.path.join(input_dir, str(example_scan))
        l = [re.match("^pred(.*).nii.gz$", x) for x in os.listdir(example_dir)]
        suffixes = [x.group(1) for x in l if x is not None]
    else:
        suffixes=[""]
        print("Warning: adjust suffixes manually in source code for v2cc")
        print("Warning: v2cc mode does not work for multiple epoch predictions")
    print("Detected suffixes:", suffixes)


    for suffix in suffixes:
        print(f"Processing {suffix}")
        args = [(suffix, input_dir, aggregate_type, icp, patient_id, scan_ids, device, v2cc, remove_source_files)
                for patient_id, scan_ids in patient_groups.items()]
        for a in tqdm(args):
            process_item_worker(a)
        # r = process_map(process_item_worker, args, max_workers=2, chunksize=5)
        # r = process_map(process_item_worker, args, max_workers=1)


    print("Done")





if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input_dir", type=str, help="Input directory containing the meshes")
    parser.add_argument("--csv-file", "-c", type=str, help="CSV file containing the patient scan mappings",
                        default="/mnt/nas/Data_Neuro/ADNI_SEG/splits/ADNI_long_val_mini.csv")
    # TODO: option to choose scan column
    parser.add_argument("--enable-icp", "-i", action="store_true", help="Enable ICP registration", default=False)
    parser.add_argument("--v2cc", action="store_true", help="Use v2cc meshes")
    parser.add_argument("--aggregate-type", "-a", type=str, default="mean", help="Type of aggregation (mean, median or both)")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--max", type=int, default=None)
    parser.add_argument("--remove_source_files", action="store_true", help="Remove source files after aggregation", default=False)

    args = parser.parse_args()

    main(args.input_dir, args.csv_file, args.enable_icp, args.aggregate_type, args.device, args.skip, args.max, args.v2cc, args.remove_source_files)
