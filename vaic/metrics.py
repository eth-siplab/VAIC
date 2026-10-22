"""The five evaluation metrics of the paper (Sec. 5.1, App. C.1).

Per IMU:
    bone_acc     100 if the estimated vertex lies on the GT bone (argmax skinning weight), else 0.
                 Bones that cannot be told apart on the torso / feet are merged (App. C.1).
    pos_err_cm   distance between estimated and GT vertex on the SMPL template mesh.
    ae_W_V       mean over t of the angle between the GT SMPL vertex orientation W_R_V(t) and
                 the calibrated sensor chain W_R_REF @ REF_R_S(t) @ S_R_V.
    ae_V_S       angle between estimated and GT mounting V_R_S.
    ae_W_REF     angle between estimated and GT reference alignment W_R_REF.
Reported as mean ± sample std over all (window, IMU) pairs.
"""
from __future__ import annotations

import pickle
from functools import lru_cache

import numpy as np
import torch

import config
from vaic.rotations import angle_deg

SMPL_JOINT_NAMES = ["pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee", "spine2",
                    "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot", "neck",
                    "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder",
                    "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hand", "right_hand"]
MERGED_BONES = [{"right_ankle", "right_foot"}, {"left_ankle", "left_foot"},
                {"spine2", "spine3", "left_collar", "right_collar"}, {"pelvis", "spine1"}]
METRICS = ["bone_acc", "pos_err_cm", "ae_W_V", "ae_V_S", "ae_W_REF"]


@lru_cache(maxsize=None)
def smpl_template() -> tuple[np.ndarray, np.ndarray]:
    """SMPL template vertices (6890, 3) and the bone of every vertex (6890,)."""
    with open(config.SMPL_MALE_FILE, "rb") as f:
        smpl = pickle.load(f, encoding="latin1")
    return np.asarray(smpl["v_template"]), np.argmax(smpl["weights"], axis=1)


def same_bone(bone_a: int, bone_b: int) -> bool:
    if bone_a == bone_b:
        return True
    names = {SMPL_JOINT_NAMES[bone_a], SMPL_JOINT_NAMES[bone_b]}
    return any(names <= group for group in MERGED_BONES)


def batch_angle_deg(R_rel: torch.Tensor) -> torch.Tensor:
    """Geodesic angles (degrees) of relative rotations (T, 3, 3)."""
    trace = R_rel.diagonal(dim1=-2, dim2=-1).sum(-1)
    return torch.rad2deg(torch.acos(torch.clamp((trace - 1) / 2.0, -1.0, 1.0)))


def imu_metrics(W_R_REF_est: torch.Tensor, V_R_S_est: torch.Tensor, vertex_est: int,
                W_R_REF_gt: torch.Tensor, V_R_S_gt: torch.Tensor, vertex_gt: int,
                REF_R_St: torch.Tensor, W_R_Vt_smpl: torch.Tensor) -> dict:
    """Metrics of one IMU. REF_R_St: (T, 3, 3) measured orientations; W_R_Vt_smpl: (T, 3, 3) GT
    SMPL orientation of the GT vertex; vertex ids are full SMPL indices."""
    v_template, vertex_bone = smpl_template()
    W_R_Vt_est = W_R_REF_est @ REF_R_St @ V_R_S_est.transpose(-1, -2)
    return {
        "bone_acc": 100.0 * same_bone(vertex_bone[vertex_est], vertex_bone[vertex_gt]),
        "pos_err_cm": float(np.linalg.norm(v_template[vertex_gt] - v_template[vertex_est]) * 100),
        "ae_W_V": batch_angle_deg(W_R_Vt_smpl.transpose(-1, -2) @ W_R_Vt_est).double().mean().item(),
        "ae_V_S": angle_deg(V_R_S_gt.transpose(-1, -2) @ V_R_S_est),
        "ae_W_REF": angle_deg(W_R_REF_gt.transpose(-1, -2) @ W_R_REF_est),
    }


def summarize(rows: list[dict]) -> dict:
    """Mean and sample std (ddof=1) of every metric over all (window, IMU) rows."""
    return {m: (float(np.mean([r[m] for r in rows])), float(np.std([r[m] for r in rows], ddof=1)))
            for m in METRICS}
