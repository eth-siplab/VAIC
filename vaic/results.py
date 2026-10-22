"""Calibration result files.

`calibration_results.hdf5` layout: /<seq_name>/<camera or 'none'>/<start>_<end>/ with
    attrs motion_name
    W_R_REF (K, 3, 3), V_R_S (K, 3, 3)   estimated calibration
    vertex_ids (K,)                      estimated SMPL vertex of every IMU
    sensor_indices (K,)                  IMU index in the dataset
    cost_matrix (K, N)                   (VAIC only) pairwise errors over the candidate vertices
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch


def save_result(path: Path, seq_name: str, motion_name: str, camera_index: int | None,
                start: int, end: int, W_R_REF: torch.Tensor, V_R_S: torch.Tensor,
                vertex_ids: np.ndarray, cost_matrix: np.ndarray | None = None) -> None:
    with h5py.File(path, "a") as f:
        grp = f.require_group(seq_name).require_group("none" if camera_index is None else str(camera_index))
        res = grp.create_group(f"{start}_{end}")
        res.attrs["motion_name"] = motion_name
        res.create_dataset("W_R_REF", data=W_R_REF.numpy())
        res.create_dataset("V_R_S", data=V_R_S.numpy())
        res.create_dataset("vertex_ids", data=np.asarray(vertex_ids, dtype=np.int64))
        res.create_dataset("sensor_indices", data=np.arange(len(vertex_ids)))
        if cost_matrix is not None:
            res.create_dataset("cost_matrix", data=cost_matrix)


def load_results(path: Path) -> list[dict]:
    """All calibrated windows of a result file."""
    windows = []
    with h5py.File(path, "r") as f:
        for seq_name in f.keys():
            for cam in f[seq_name].keys():
                for rng, res in f[seq_name][cam].items():
                    start, end = (int(x) for x in rng.split("_"))
                    windows.append({
                        "seq_name": seq_name, "camera_index": None if cam == "none" else int(cam),
                        "start_index": start, "end_index": end, "motion_name": res.attrs["motion_name"],
                        "W_R_REF": torch.from_numpy(res["W_R_REF"][:]),
                        "V_R_S": torch.from_numpy(res["V_R_S"][:]),
                        "vertex_ids": res["vertex_ids"][:], "sensor_indices": res["sensor_indices"][:],
                    })
    return windows
