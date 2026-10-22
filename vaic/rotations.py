"""Rotation utilities on SO(3).

Conventions: `A_R_B` rotates vectors from frame B into frame A. Axis-angle vectors are the
Lie-algebra coordinates `omega = Log(R)`.

`axis_angle_to_rotation_matrix` and `rotation_matrix_to_axis_angle` are taken from the
`articulate` toolkit of PIP (https://github.com/Xinyu-Yi/PIP, GPL-3.0) to keep numerics identical to the
experiments in the paper.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch


def _skew(x: torch.Tensor) -> torch.Tensor:
    """Skew-symmetric matrices [x]_x for vectors x of shape (B, 3)."""
    x = x.view(-1, 3)
    zeros = torch.zeros(x.shape[0], device=x.device)
    return torch.stack((zeros, -x[:, 2], x[:, 1],
                        x[:, 2], zeros, -x[:, 0],
                        -x[:, 1], x[:, 0], zeros), dim=1).view(-1, 3, 3)


def axis_angle_to_rotation_matrix(a: torch.Tensor) -> torch.Tensor:
    """Axis-angle (any shape reshapeable to (B, 3)) -> rotation matrices (B, 3, 3)."""
    angle = a.view(-1, 3).norm(dim=-1, keepdim=True)
    axis = a.view(-1, 3) / angle
    axis[torch.isnan(axis) | torch.isinf(axis)] = 0
    i_cube = torch.eye(3, device=a.device).expand(angle.shape[0], 3, 3)
    c, s = angle.cos().view(-1, 1, 1), angle.sin().view(-1, 1, 1)
    return c * i_cube + (1 - c) * torch.bmm(axis.view(-1, 3, 1), axis.view(-1, 1, 3)) + s * _skew(axis)


def rotation_matrix_to_axis_angle(r: torch.Tensor) -> torch.Tensor:
    """Rotation matrices (any shape reshapeable to (B, 3, 3)) -> axis-angle (B, 3)."""
    result = [cv2.Rodrigues(m)[0] for m in r.clone().detach().cpu().view(-1, 3, 3).numpy()]
    return torch.from_numpy(np.stack(result)).float().squeeze(-1).to(r.device)


def rotation_matrix_to_axis_angle_batched(R: torch.Tensor) -> torch.Tensor:
    """Vectorised rotation matrices (B, 3, 3) -> axis-angle (B, 3) on R's device, computed in float64.
    Used for the DS_VAIC virtual IMUs (the paper data was computed with it)."""
    R64 = R.to(dtype=torch.float64)
    trace = R64[:, 0, 0] + R64[:, 1, 1] + R64[:, 2, 2]
    theta = torch.acos(torch.clamp((trace - 1) / 2, -1.0, 1.0))
    w = torch.stack([R64[:, 2, 1] - R64[:, 1, 2], R64[:, 0, 2] - R64[:, 2, 0], R64[:, 1, 0] - R64[:, 0, 1]], dim=1)
    sin_theta = torch.norm(w, dim=1) / 2
    axis_angle = w / (2 * sin_theta).unsqueeze(1).clamp(min=1e-8) * theta.unsqueeze(1)
    small = theta < 1e-4
    axis_angle[small] = 0.5 * w[small]
    return axis_angle.to(dtype=torch.float32)


def log_map(R: torch.Tensor) -> torch.Tensor:
    """Differentiable SO(3) logarithm. R: (B, 3, 3) or (3, 3) -> omega: (B, 3) or (3,)."""
    is_single = R.dim() == 2
    if is_single:
        R = R.unsqueeze(0)
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    theta = torch.acos(((trace - 1) / 2).clamp(-1.0, 1.0))
    # theta / (2 sin(theta)) -> 0.5 for small angles
    scale = torch.where(torch.sin(theta) < 1e-6, 0.5, theta / (2 * torch.sin(theta)))
    skew = scale.unsqueeze(-1).unsqueeze(-1) * (R - R.transpose(-2, -1))
    omega = torch.stack([skew[:, 2, 1], skew[:, 0, 2], skew[:, 1, 0]], dim=1)
    return omega.squeeze(0) if is_single else omega


def exp_map(omega: torch.Tensor) -> torch.Tensor:
    """SO(3) exponential (Rodrigues). omega: (B, 3) or (3,) -> R: (B, 3, 3) or (3, 3)."""
    is_single = omega.dim() == 1
    if is_single:
        omega = omega.unsqueeze(0)
    theta = torch.linalg.norm(omega, dim=1, keepdim=True)
    k = omega / (theta + 1e-9)
    zeros = torch.zeros_like(k[:, 0])
    K = torch.stack([zeros, -k[:, 2], k[:, 1],
                     k[:, 2], zeros, -k[:, 0],
                     -k[:, 1], k[:, 0], zeros], dim=1).view(-1, 3, 3)
    eye = torch.eye(3, device=omega.device, dtype=omega.dtype).expand(omega.shape[0], 3, 3)
    R = eye + torch.sin(theta).unsqueeze(-1) * K + (1 - torch.cos(theta)).unsqueeze(-1) * (K @ K)
    return R.squeeze(0) if is_single else R


def angle_deg(R_rel: torch.Tensor) -> float:
    """Geodesic angle (degrees) of a single relative rotation (3, 3)."""
    trace = torch.trace(R_rel)
    return torch.rad2deg(torch.acos(torch.clamp((trace - 1) / 2.0, -1.0, 1.0))).item()


def karcher_mean(Rs: torch.Tensor, tol: float = 1e-9, max_iters: int = 50) -> torch.Tensor:
    """Intrinsic (Karcher) mean of rotations Rs: (T, 3, 3) -> (3, 3).

    Initialised with the chordal (SVD-projected arithmetic) mean, then refined by averaging
    the tangent vectors Log(R_mean^T R_t) and stepping along the geodesic.
    """
    U, _, Vt = torch.linalg.svd(Rs.mean(dim=0))
    R_mean = U @ Vt
    if torch.linalg.det(R_mean) < 0:
        U[:, -1] *= -1
        R_mean = U @ Vt
    for _ in range(max_iters):
        delta = log_map(torch.matmul(R_mean.transpose(-2, -1), Rs)).mean(dim=0)
        if torch.linalg.norm(delta) < tol:
            break
        R_mean = R_mean @ exp_map(delta)
    return R_mean
