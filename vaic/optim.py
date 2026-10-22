"""Batched Levenberg-Marquardt solver for the sensor-to-vertex mounting V_R_S (paper Sec. 4.2).

For every (IMU i, vertex j) pair we optimise the yaw-dominant tangent-frame rotation
`T_R_S = Exp(omega)`, with `V_R_S = V_R_T @ T_R_S`, minimising

    E = E_ori + lambda_normal * E_norm + lambda_acc * E_acc

    E_ori  = sum_t || Log( V(t)_R_V(t-1) @ V_R_S @ S(t-1)_R_S(t) @ S_R_V ) ||^2          (Eq. 4)
    E_norm = || omega_xy ||^2                                                          (Eq. 5)
    E_acc  = sum_t || a_W(t) - W_R_V(t) @ V_R_S @ S(t)_R_REF @ a_REF(t) ||^2             (Eq. 6)

All K x N pairs are solved in parallel on the GPU, starting from the identity, with a
forward-difference Jacobian (App. B.1).
"""
from __future__ import annotations

import numpy as np
import torch

from vaic.rotations import axis_angle_to_rotation_matrix, exp_map, log_map


def solve_mountings(imu_acc: torch.Tensor, imu_rot_aa: torch.Tensor,
                    vertex_acc: torch.Tensor, vertex_rot_aa: torch.Tensor, V_R_T: torch.Tensor,
                    lambda_acc: float, lambda_normal: float,
                    max_iter: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Optimise V_R_S for all IMU-vertex pairs.

    Args:
        imu_acc: (K, T, 3) IMU accelerations in REF.
        imu_rot_aa: (K, T, 3) IMU orientations REF_R_S(t), axis-angle.
        vertex_acc: (N, T, 3) virtual vertex accelerations in W.
        vertex_rot_aa: (N, T, 3) virtual vertex orientations W_R_V(t), axis-angle.
        V_R_T: (N, 3, 3) tangent frame of every candidate vertex.
    Returns:
        cost: (K, N) residual error of every pair.
        V_R_S: (K, N, 3, 3) optimal mounting of every pair.
    """
    device = imu_acc.device
    K, T, _ = imu_acc.shape
    N = vertex_acc.shape[0]
    B_sz = K * N

    # Flatten pairs to a batch: pair b = i * N + j.
    imu_rot_b = imu_rot_aa.unsqueeze(1).expand(-1, N, -1, -1).reshape(B_sz, T, 3)
    vertex_rot_b = vertex_rot_aa.unsqueeze(0).expand(K, -1, -1, -1).reshape(B_sz, T, 3)
    imu_acc_b = imu_acc.unsqueeze(1).expand(-1, N, -1, -1).reshape(B_sz, T, 3)
    vertex_acc_b = vertex_acc.unsqueeze(0).expand(K, -1, -1, -1).reshape(B_sz, T, 3)
    V_R_T_b = V_R_T.unsqueeze(0).expand(K, -1, -1, -1).reshape(B_sz, 3, 3)

    W_R_Vt = axis_angle_to_rotation_matrix(vertex_rot_b.reshape(-1, 3)).reshape(B_sz, T, 3, 3)
    REF_R_St = axis_angle_to_rotation_matrix(imu_rot_b.reshape(-1, 3)).reshape(B_sz, T, 3, 3)
    St_R_REF = REF_R_St.transpose(-2, -1)
    Vt_R_Vprev = torch.matmul(W_R_Vt[:, 1:].transpose(-2, -1), W_R_Vt[:, :-1])
    Sprev_R_St = torch.matmul(REF_R_St[:, :-1].transpose(-2, -1), REF_R_St[:, 1:])

    def residuals(omega: torch.Tensor) -> torch.Tensor:
        V_R_S = torch.matmul(V_R_T_b, exp_map(omega)).unsqueeze(1)
        S_R_V = V_R_S.transpose(-2, -1)
        inner = torch.matmul(Vt_R_Vprev, torch.matmul(V_R_S, torch.matmul(Sprev_R_St, S_R_V)))
        r_ori = log_map(inner.reshape(-1, 3, 3)).reshape(B_sz, -1)

        W_R_REF = torch.matmul(W_R_Vt, torch.matmul(V_R_S, St_R_REF))
        a_W = torch.matmul(W_R_REF, imu_acc_b.unsqueeze(-1)).squeeze(-1)
        r_acc = ((vertex_acc_b - a_W) * np.sqrt(lambda_acc)).reshape(B_sz, -1)

        r_norm = omega[:, :2] * np.sqrt(lambda_normal)
        return torch.cat([r_ori, r_acc, r_norm], dim=1)

    # Initialise V_R_S = I, i.e. T_R_S = T_R_V.
    omega = log_map(V_R_T_b.transpose(-2, -1))
    damping = torch.ones(B_sz, device=device) * 1e-2
    res = residuals(omega)
    error = torch.sum(res ** 2, dim=1)
    eye = torch.eye(3, device=device).unsqueeze(0).expand(B_sz, -1, -1)

    for _ in range(max_iter):
        eps = 1e-5
        J = torch.stack([(residuals(omega + eps * torch.eye(3, device=device)[d]) - res) / eps
                         for d in range(3)], dim=2)
        JtJ = torch.bmm(J.transpose(1, 2), J)
        Jtr = torch.bmm(J.transpose(1, 2), res.unsqueeze(-1)).squeeze(-1)
        step, info = torch.linalg.solve_ex(JtJ + eye * damping.view(B_sz, 1, 1), -Jtr)
        if torch.any(info != 0):
            # singular system somewhere in the batch: increase damping and retry
            damping = torch.clamp(damping * 100.0, min=1e-4)
            continue

        omega_new = omega + step
        res_new = residuals(omega_new)
        error_new = torch.sum(res_new ** 2, dim=1)
        improved = error_new < error
        omega = torch.where(improved.unsqueeze(-1), omega_new, omega)
        res = torch.where(improved.unsqueeze(-1), res_new, res)
        error = torch.where(improved, error_new, error)
        damping = torch.clamp(torch.where(improved, damping * 0.1, damping * 10.0), 1e-7, 1e7)

    V_R_S = torch.matmul(V_R_T_b, exp_map(omega)).reshape(K, N, 3, 3)
    return error.reshape(K, N), V_R_S
