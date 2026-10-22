"""VAIC (Alg. 1, lines 4-11) and the T-Pose baseline."""
from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

import config
from vaic.optim import solve_mountings
from vaic.rotations import axis_angle_to_rotation_matrix, karcher_mean


def vaic(inputs: dict, fixed: bool, lambda_acc: float = config.LAMBDA_ACC,
         lambda_normal: float = config.LAMBDA_NORMAL, max_iter: int = config.MAX_ITER) -> dict:
    """Calibrate all IMUs of one window.

    1. Optimise the mounting V_R_S for every IMU-vertex pair -> cost matrix (K, N).
    2. Assign IMUs to vertices with the Hungarian algorithm (or use the GT vertices if `fixed`).
    3. W_R_REF = Karcher mean over t of W_R_V(t) @ V_R_S @ S(t)_R_REF                       (Eq. 7-8)

    Returns dict with W_R_REF (K, 3, 3), V_R_S (K, 3, 3), vertex_idx (K,) into the candidate set
    and cost_matrix (K, N).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cost, V_R_S_all = solve_mountings(
        inputs["imu_acc"].permute(1, 0, 2).to(device), inputs["imu_rot_aa"].permute(1, 0, 2).to(device),
        inputs["vacc_glb"].permute(1, 0, 2).to(device), inputs["vrot_aa_glb"].permute(1, 0, 2).to(device),
        config.V_R_T[config.VERTEX_MASK].to(device), lambda_acc, lambda_normal, max_iter)
    cost = cost.detach().cpu().numpy()
    V_R_S_all = V_R_S_all.detach().cpu()

    if fixed:
        imu_idx, vertex_idx = np.arange(cost.shape[0]), np.array(inputs["gt_vertex_idx"])
    else:
        imu_idx, vertex_idx = linear_sum_assignment(cost)

    W_R_REF, V_R_S = [], []
    for i, j in zip(imu_idx, vertex_idx):
        W_R_Vt = axis_angle_to_rotation_matrix(inputs["vrot_aa_glb"][:, j, :])
        REF_R_St = axis_angle_to_rotation_matrix(inputs["imu_rot_aa"][:, i, :])
        W_R_REF.append(karcher_mean(torch.matmul(W_R_Vt, torch.matmul(V_R_S_all[i, j], REF_R_St.transpose(-2, -1)))))
        V_R_S.append(V_R_S_all[i, j])
    return {"W_R_REF": torch.stack(W_R_REF), "V_R_S": torch.stack(V_R_S),
            "vertex_idx": np.asarray(vertex_idx), "cost_matrix": cost}


# REF is the IMU's East-North-Up estimate (Z-up); W is Y-up. Ideal alignment assumed by T-Pose.
W_R_REF_ENU = torch.tensor([[-1.0, 0, 0], [0, 0, 1], [0, 1, 0]])


def tpose(REF_R_S: torch.Tensor) -> dict:
    """T-Pose baseline: assume W_R_REF = W_R_REF_ENU and W_R_V = I during the T-pose.

    V_R_S = W_R_V^T @ W_R_REF @ REF_R_S_tpose = W_R_REF @ REF_R_S_tpose, with REF_R_S_tpose the
    Karcher mean over the T-pose frames.
    REF_R_S: (T_tpose, K, 3, 3). Returns W_R_REF (K, 3, 3) and V_R_S (K, 3, 3).
    """
    K = REF_R_S.shape[1]
    REF_R_S_mean = torch.stack([karcher_mean(REF_R_S[:, i]) for i in range(K)])
    REF_R_W = W_R_REF_ENU.transpose(-1, -2).unsqueeze(0).expand(K, -1, -1)
    S_R_V = torch.matmul(REF_R_S_mean.transpose(-1, -2), REF_R_W)
    return {"W_R_REF": REF_R_W.transpose(-1, -2), "V_R_S": S_R_V.transpose(-1, -2)}
