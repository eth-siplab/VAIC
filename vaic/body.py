"""Virtual vertex IMUs from SMPL / SMPL-X motion (paper Sec. 4.1 and App. B).

Vertex orientations `W_R_V` are the linear-blend-skinned joint rotations of every vertex
(`vert_rot`, Eq. 9-10), provided by our modified `smplx` package via `compute_rotations=True`.
Vertex accelerations are second-order finite differences of the vertex positions; `synthesize_acc`
is adapted from PIP (https://github.com/Xinyu-Yi/PIP, GPL-3.0).
"""
from __future__ import annotations

from functools import lru_cache

import smplx
import torch

import config
from vaic.rotations import rotation_matrix_to_axis_angle


def synthesize_acc(v: torch.Tensor, smooth_n: int = 4, fps: float = config.FPS) -> torch.Tensor:
    """Gravity-free accelerations (T, N, 3) from positions v (T, N, 3) by finite differences.

    Inner frames use a wider stencil of `smooth_n` frames for smoothing; the borders use the
    1-frame stencil, and the first/last frame are zero.
    """
    factor = fps ** 2
    acc = torch.stack([(v[i] + v[i + 2] - 2 * v[i + 1]) * factor for i in range(0, v.shape[0] - 2)])
    acc = torch.cat((torch.zeros_like(acc[:1]), acc, torch.zeros_like(acc[:1])))
    acc[smooth_n:-smooth_n] = torch.stack(
        [(v[i] + v[i + smooth_n * 2] - 2 * v[i + smooth_n]) * factor / smooth_n ** 2
         for i in range(0, v.shape[0] - smooth_n * 2)])
    return acc


def project_to_so3(R: torch.Tensor) -> torch.Tensor:
    """Closest rotation matrix (SVD, Eq. 11-12), applied to the LBS-blended vertex rotations, which are not
    orthonormal."""
    U, _, Vt = torch.linalg.svd(R)
    S = torch.eye(3, device=R.device).expand_as(U).clone()
    S[..., 2, 2] = torch.det(torch.matmul(U, Vt))
    return torch.matmul(U, torch.matmul(S, Vt))


@lru_cache(maxsize=None)
def smplx_model(gender: str) -> torch.nn.Module:
    return smplx.create(str(config.BODY_MODEL_DIR), model_type="smplx", num_betas=10,
                        gender=gender, use_pca=False)


@lru_cache(maxsize=None)
def smpl_model(gender: str = "male") -> torch.nn.Module:
    return smplx.create(str(config.SMPL_FILES[gender]), "smpl", num_betas=10, gender=gender,
                        use_pca=False, flat_hand_mean=True, ext="pkl")


def smplx_vertex_imu(poses_aa: torch.Tensor, trans: torch.Tensor, betas: torch.Tensor,
                     smplx_vertices: list[int], gender: str = "neutral",
                     fps: float = config.FPS) -> tuple[torch.Tensor, torch.Tensor]:
    """Virtual IMU signals of selected vertices from an SMPL-X body motion (video estimate).

    Args:
        poses_aa: (T, 66) root + 21 body joints, axis-angle.
        trans: (T, 3) root translation.
        betas: (10,) shape parameters.
        smplx_vertices: SMPL-X vertex indices to synthesize.
    Returns:
        vrot_aa_glb: (T, N, 3) vertex orientations W_R_V (axis-angle).
        vacc_glb: (T, N, 3) gravity-free vertex accelerations in W.
    """
    T = poses_aa.shape[0]
    if betas.dim() == 1:
        betas = betas.unsqueeze(0)

    def zeros(n: int) -> torch.Tensor:
        return torch.zeros(T, n)

    out = smplx_model(gender)(
        global_orient=poses_aa[:, :3], body_pose=poses_aa[:, 3:], transl=trans,
        betas=betas.expand(T, -1), expression=zeros(10), jaw_pose=zeros(3),
        leye_pose=zeros(3), reye_pose=zeros(3), left_hand_pose=zeros(45), right_hand_pose=zeros(45),
        return_verts=True, return_full_pose=True, compute_rotations=True)

    vacc_glb = synthesize_acc(out.vertices[:, smplx_vertices, :], fps=fps)
    W_R_V = project_to_so3(out.vert_rot[:, smplx_vertices])
    vrot_aa_glb = rotation_matrix_to_axis_angle(W_R_V.reshape(-1, 3, 3)).reshape(T, len(smplx_vertices), 3)
    return vrot_aa_glb, vacc_glb
