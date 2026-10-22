"""Loading calibration windows from the preprocessed HDF5 files and filtering invalid ones.

HDF5 layout (one group per recorded sequence, attribute `motion_name`):
    smpl_pose_aa (T, 24, 3), smpl_tran (T, 3)      GT SMPL motion (mocap), world frame W (Y-up)
    vrot_aa_glb, vacc_glb (T, 372, 3)              GT virtual vertex IMUs of the candidate vertices
    ref_R_S (T, K, 3, 3), ref_a (T, K, 3)          raw IMU orientation / acceleration in REF
    W_R_ref (K, 3, 3), B_R_S_aligned (K, 3, 3)     GT calibration W_R_REF and V_R_S
    imu_vertex_ids (K,)                            GT SMPL vertex of every IMU
    pose_smplx_aa_cam{c} (T, 66), trans_smplx_cam{c} (T, 3), shape_smplx_cam{c} (10,),
    detection_cam{c} (T, 1)                        SMPL-X estimated from video of camera c
    events_json                                    (DS_VAIC only) protocol events, e.g. DYNAMIC_START
    smpl_betas (10,), attribute gender             (DS_VAIC only) GT body shape, used by preprocess.py
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.spatial.transform import Rotation

import config
from vaic.body import smplx_vertex_imu
from vaic.rotations import axis_angle_to_rotation_matrix, rotation_matrix_to_axis_angle

FRAME_KEYS = ["smpl_pose_aa", "smpl_tran", "vrot_aa_glb", "vacc_glb", "ref_R_S", "ref_a"]
STATIC_KEYS = ["W_R_ref", "B_R_S_aligned", "imu_vertex_ids"]
CAM_FRAME_KEYS = ["pose_smplx_aa_cam", "trans_smplx_cam", "detection_cam"]


def find_event_frame(events: list[dict], prefix: str, fps: float = config.FPS) -> int | None:
    """Frame of the first protocol event whose label starts with `prefix`."""
    for evt in events:
        if " ".join(evt["label"].split()).startswith(" ".join(prefix.split())):
            return int(round(evt["t"] * fps))
    return None


def read_events(grp: h5py.Group) -> list[dict] | None:
    if "events_json" not in grp:
        return None
    raw = grp["events_json"][()]
    return json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)


def masked_vertex_index(smpl_vertex_ids: torch.Tensor) -> list[int]:
    """Map SMPL vertex ids to indices into the candidate set config.VERTEX_MASK."""
    mask = config.VERTEX_MASK.tolist()
    return [mask.index(int(v)) for v in smpl_vertex_ids]


class CalibrationWindows(torch.utils.data.Dataset):
    """All calibration windows of length `seq_len` of a dataset, one per (sequence, camera, chunk).

    Sequences are cut into consecutive chunks of `seq_len` frames. For DS_VAIC only the dynamic
    calibration block (DYNAMIC_START .. DYNAMIC_END) is chunked; the validation block that follows
    is kept for evaluation. The first `CUT_OFF_FRONT` frames of every chunk are skipped.
    """

    def __init__(self, data_path: Path, seq_len: int) -> None:
        self.data_path = data_path
        self.seq_len = seq_len
        self.windows: list[tuple[str, int, int | None]] = []
        with h5py.File(data_path, "r") as f:
            for seq_name in sorted(f.keys()):
                grp = f[seq_name]
                n_frames = grp["smpl_pose_aa"].shape[0]
                cam_ids = [int(k.split("cam")[-1][0]) for k in grp.keys() if "cam" in k and "shape" in k]
                start, length = 0, n_frames
                events = read_events(grp)
                if events is not None:
                    start = find_event_frame(events, "DYNAMIC_START")
                    length = find_event_frame(events, "DYNAMIC_END") - start
                chunk_starts = [start + i * seq_len for i in range(length // seq_len)]
                for cam_idx in cam_ids:
                    self.windows.extend((seq_name, s, cam_idx) for s in chunk_starts)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict:
        seq_name, chunk_start, cam_idx = self.windows[idx]
        start = chunk_start + config.CUT_OFF_FRONT
        end = start + self.seq_len
        item = {"seq_name": seq_name, "camera_index": cam_idx, "start_index": start, "end_index": end}
        with h5py.File(self.data_path, "r") as f:
            grp = f[seq_name]
            item["motion_name"] = grp.attrs["motion_name"]
            for key in FRAME_KEYS + [k + str(cam_idx) for k in CAM_FRAME_KEYS]:
                item[key] = torch.tensor(grp[key][start:end])
            for key in STATIC_KEYS + [f"shape_smplx_cam{cam_idx}"]:
                item[key] = torch.tensor(grp[key][:])
        return item


def is_excluded(window: dict) -> bool:
    """Windows excluded from the evaluation for known dataset issues."""
    if window["camera_index"] in config.BROKEN_TC_CAMERAS:
        return True
    if window["motion_name"] in config.IGNORED_TC_SEQUENCES:
        return True
    return (window["motion_name"], window["start_index"]) in config.GT_ISSUES


def has_movement(window: dict, tran_std_thresh: float = 0.1, pose_std_thresh: float = 0.1) -> bool:
    """Motion filter (App. B.2): ground-plane root translation std > 0.1 m and mean pose std > 0.1 rad,
    for both the GT motion and the video estimate."""
    c = window["camera_index"]
    for tran, pose in [(window["smpl_tran"], window["smpl_pose_aa"]),
                       (window[f"trans_smplx_cam{c}"], window[f"pose_smplx_aa_cam{c}"])]:
        if torch.all(torch.std(tran, dim=0)[[0, 2]] < tran_std_thresh):
            return False
        if torch.mean(torch.std(pose, dim=0)) < pose_std_thresh:
            return False
    return True


def align_trajectories(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rigid (R, t) minimising ||R source + t - target|| (Kabsch)."""
    centroid_source, centroid_target = source.mean(axis=0), target.mean(axis=0)
    H = (source - centroid_source).T @ (target - centroid_target)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T
    return R, centroid_target - R @ centroid_source


def apply_alignment(trans: np.ndarray, root_orient_aa: np.ndarray, R: np.ndarray,
                    t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply a rigid world transform to root translation (N, 3) and root orientation (N, 3)."""
    root_R = np.matmul(R, Rotation.from_rotvec(root_orient_aa).as_matrix())
    return (R @ trans.T).T + t, Rotation.from_matrix(root_R).as_rotvec()


def align_video_to_gt(window: dict) -> None:
    """Express the video SMPL-X motion in the mocap world frame W (rigid trajectory alignment).

    Video-based mesh recovery has an unknown absolute heading; it is aligned to the GT only to
    make evaluation in W possible (App. C).
    """
    c = window["camera_index"]
    trans = window[f"trans_smplx_cam{c}"]
    pose = window[f"pose_smplx_aa_cam{c}"]
    source_pose = pose.cpu().numpy()
    R, t = align_trajectories(trans.cpu().numpy(), window["smpl_tran"].cpu().numpy())
    trans_aligned, root_aligned = apply_alignment(trans.cpu().numpy(), source_pose[:, :3], R, t)
    source_pose[:, :3] = root_aligned
    window[f"trans_smplx_cam{c}"] = torch.from_numpy(trans_aligned).to(trans)
    window[f"pose_smplx_aa_cam{c}"] = torch.from_numpy(source_pose).to(pose)


def has_video_failure(window: dict) -> bool:
    """Flag windows where video mesh recovery failed although a person was detected:
    a static, flying-off or upside-down/rotated body compared to the GT motion."""
    c = window["camera_index"]
    gt_pose, gt_trans = window["smpl_pose_aa"].float(), window["smpl_tran"].float()
    cam_pose, cam_trans = window[f"pose_smplx_aa_cam{c}"].float(), window[f"trans_smplx_cam{c}"].float()

    gt_vel_vec = gt_trans[1:] - gt_trans[:-1]
    cam_vel_vec = cam_trans[1:] - cam_trans[:-1]
    gt_vel, cam_vel = gt_vel_vec.norm(dim=1), cam_vel_vec.norm(dim=1)
    if (gt_vel - cam_vel).std().item() > 0.0033:
        return True

    # static prediction while the subject moves
    static_mismatch = (gt_vel > 0.005) & (cam_vel < 0.001)
    if static_mismatch.float().mean().item() > 0.15:
        return True

    # prediction flies off
    vel_diff = (gt_vel_vec - cam_vel_vec).norm(dim=1)
    if (vel_diff > 0.013).float().mean().item() > 0.10:
        return True

    # large root orientation error
    gt_R = axis_angle_to_rotation_matrix(gt_pose[:, 0].contiguous())
    cam_R = axis_angle_to_rotation_matrix(cam_pose[:, :3].contiguous())
    R_diff = torch.matmul(gt_R, cam_R.transpose(1, 2))
    cos_angle = ((R_diff.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1.0, 1.0)
    return (torch.rad2deg(torch.acos(cos_angle)) > 30.0).float().mean().item() > 0.10


def is_valid(window: dict) -> bool:
    """Motion filter + video failure filters (App. B.2). Aligns the video motion as side effect."""
    if not has_movement(window):
        return False
    if not torch.all(window[f"detection_cam{window['camera_index']}"]):
        return False
    align_video_to_gt(window)
    return not has_video_failure(window)


def prepare_inputs(window: dict, use_gt_smpl: bool) -> dict:
    """Inputs of the pairwise optimisation for one window.

    Returns:
        imu_acc: (T, K, 3) measured accelerations in REF (with gravity).
        imu_rot_aa: (T, K, 3) measured orientations REF_R_S(t), axis-angle.
        vacc_glb: (T, N, 3) virtual vertex accelerations in W, gravity added (proper acceleration).
        vrot_aa_glb: (T, N, 3) virtual vertex orientations W_R_V(t), axis-angle.
        gt_vertex_idx: (K,) index of the GT vertex of every IMU in the candidate set.
    """
    T, K = window["ref_a"].shape[:2]
    # Round trip through axis-angle: turns the stored float32 matrices into exact rotation matrices.
    REF_R_S = axis_angle_to_rotation_matrix(rotation_matrix_to_axis_angle(window["ref_R_S"].reshape(-1, 3, 3)))
    if use_gt_smpl:
        vrot_aa_glb, vacc_glb = window["vrot_aa_glb"], window["vacc_glb"]
    else:
        c = window["camera_index"]
        smplx_vertices = [config.SMPL_TO_SMPLX_VERTEX_MAP[i.item()] for i in config.VERTEX_MASK]
        with torch.no_grad():
            vrot_aa_glb, vacc_glb = smplx_vertex_imu(
                window[f"pose_smplx_aa_cam{c}"], window[f"trans_smplx_cam{c}"],
                window[f"shape_smplx_cam{c}"], smplx_vertices)
    return {
        "imu_acc": window["ref_a"],
        "imu_rot_aa": rotation_matrix_to_axis_angle(REF_R_S).reshape(T, K, 3),
        "vacc_glb": vacc_glb + config.GRAVITY_Y_UP.to(vacc_glb),
        "vrot_aa_glb": vrot_aa_glb,
        "gt_vertex_idx": masked_vertex_index(window["imu_vertex_ids"]),
    }
