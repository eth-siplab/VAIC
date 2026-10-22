"""Build the HDF5 files used by calibrate.py / evaluate.py.

--ds TC (default): TotalCapture. Inputs (defaults: the locations used by data_preprocessing/download_*.sh)
    --tc_imu     raw TotalCapture IMU data:    <tc_imu>/S1_imu/s1/s1_acting1_Xsens.sensors
                                               (+ s1_acting1_calib_imu_ref.txt, s1_acting1_calib_imu_bone.txt)
    --tc_gt      TotalCapture SMPL GT poses (DIP-IMU format, 60 fps):  <tc_gt>/s1_acting1.pkl  ('gt': (T, 72))
    --tc_pos     official TotalCapture GT positions (root translation): <tc_pos>/S1/acting1/gt_skel_gbl_pos.txt
    --video      our SMPL-X video estimates, one folder per camera:
                 <video>/S1/acting1/TC_S1_acting1_cam<c>/{subject-1.smpl, results_detected.pkl}

    python preprocess.py [--tc_imu ...] [--tc_gt ...] [--tc_pos ...] [--video ...] [--out data/tc.h5]

--ds VAIC: DS_VAIC. Adds the GT virtual vertex IMUs (`vrot_aa_glb`, `vacc_glb` of every candidate vertex,
from the GT SMPL motion with the participant's shape and gender) to the downloaded data/ds_vaic_raw.h5.
On a GPU this reproduces the paper data exactly; on the CPU orientations near 180° differ slightly (Tab. 3 unchanged).

    python preprocess.py --ds VAIC [--vaic_raw data/ds_vaic_raw.h5] [--out data/ds_vaic.h5]

Frames: W is the TotalCapture world (Y-up), REF the per-IMU reference frame (ENU, Z-up), S the sensor.
The GT mounting V_R_S is defined such that sensor and SMPL vertex orientation agree in the first frame.
The TotalCapture readers are adapted from PIP's preprocess.py (https://github.com/Xinyu-Yi/PIP, GPL-3.0).
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import h5py
import joblib
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation
from tqdm import tqdm

import config
from vaic.body import smpl_model, synthesize_acc
from vaic.data import align_trajectories, apply_alignment
from vaic.rotations import rotation_matrix_to_axis_angle, rotation_matrix_to_axis_angle_batched

INCH_TO_M = 0.0254
W_R_TCWORLD = Rotation.from_euler("y", 180, degrees=True).as_matrix()   # TC calibration files -> W
NUM_SENSORS = len(config.TC_SENSOR_NAMES)


# --- TotalCapture readers ---------------------------------------------------------------------
def read_xsens(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Raw IMU stream -> REF_R_S (T, 13, 3, 3) and sensor-frame accelerations S_a (T, 13, 3)."""
    lines = path.read_text().splitlines(keepends=True)
    del lines[0]
    del lines[::NUM_SENSORS + 1]           # per-frame header lines
    data = np.array([[float(x) for x in line.split()[1:]] for line in lines])
    quat_wxyz, S_a = data[:, :4], data[:, 4:]
    REF_R_S = np.array([Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix() for q in quat_wxyz])
    n_frames = len(REF_R_S) // NUM_SENSORS
    return REF_R_S.reshape(n_frames, NUM_SENSORS, 3, 3), S_a.reshape(n_frames, NUM_SENSORS, 3)


def read_calibration(path: Path) -> np.ndarray:
    """TotalCapture calibration file (quaternions x y z w) -> rotations (13, 3, 3)."""
    lines = path.read_text().splitlines()[1:]
    return np.array([Rotation.from_quat([float(x) for x in line.split()[1:]]).as_matrix() for line in lines])


def read_translation(path: Path) -> torch.Tensor:
    """Root translation (T, 3) in W from the 'Spine' marker of gt_skel_gbl_pos.txt, relative to frame 0."""
    lines = path.read_text().split("\n")
    spine = lines[0].split("\t").index("Spine")
    pos = [[[float(v) for v in p.split(" ")] for p in line.split("\t")[:-1]] for line in lines[1:] if line]
    pos = torch.tensor(pos)[:, spine] * INCH_TO_M
    pos[:, 0].neg_()
    pos[:, 2].neg_()
    return pos - pos[:1]


# --- GT ---------------------------------------------------------------------------------------
def smooth_motion(pose_aa: np.ndarray, trans: np.ndarray, sigma: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian smoothing of the GT motion against marker jitter (quaternion space for rotations)."""
    trans = gaussian_filter1d(trans, sigma=sigma, axis=0, mode="nearest")
    pose_smooth = np.zeros_like(pose_aa)
    for j in range(pose_aa.shape[1]):
        quats = Rotation.from_rotvec(pose_aa[:, j]).as_quat()
        for t in range(1, len(quats)):      # keep the quaternion hemisphere continuous
            if np.dot(quats[t], quats[t - 1]) < 0:
                quats[t] = -quats[t]
        quats = gaussian_filter1d(quats, sigma=sigma, axis=0, mode="nearest")
        quats /= np.linalg.norm(quats, axis=1, keepdims=True)
        pose_smooth[:, j] = Rotation.from_quat(quats).as_rotvec()
    return pose_smooth, trans


def smpl_vertex_rotations(pose: torch.Tensor, trans: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """SMPL (male, mean shape) vertex positions (T, 6890, 3) and vertex orientations W_R_V (T, 6890, 3, 3)."""
    with torch.no_grad():
        out = smpl_model()(global_orient=pose[:, 0], body_pose=pose[:, 1:].reshape(-1, 69),
                           transl=trans, compute_rotations=True)
    return out.vertices, out.vert_rot


def gt_mounting(W_R_REF: np.ndarray, REF_R_S0: np.ndarray, W_R_V0: np.ndarray) -> np.ndarray:
    """GT V_R_S per sensor from the first frame: V_R_S = (S0_R_W @ W_R_V0)^T."""
    return np.stack([np.dot(np.dot(W_R_REF[s], REF_R_S0[s]).T, W_R_V0[s]).T for s in range(len(W_R_REF))])


# --- Video ------------------------------------------------------------------------------------
def load_video(cam_dir: Path, n_frames: int, gt_trans: np.ndarray) -> dict | None:
    """SMPL-X estimate of one camera, rigidly aligned to the GT trajectory. None if too short."""
    smpl = np.load(cam_dir / "subject-1.smpl", allow_pickle=True)
    body_pose = smpl["bodyPose"]
    if body_pose.shape[0] < n_frames:
        return None
    pose_aa = np.concatenate([body_pose[:, :1].reshape(-1, 3), body_pose[:, 1:].reshape(-1, 63)], axis=1)[:n_frames]
    trans = smpl["bodyTranslation"][:n_frames]
    detected = np.array(joblib.load(cam_dir / "results_detected.pkl")[1]["detected"])[:n_frames]
    mask = detected.astype(bool)
    if np.sum(mask) > 10:
        R, t = align_trajectories(trans[mask], gt_trans[mask])
        trans[:], pose_aa[:, :3] = apply_alignment(trans, pose_aa[:, :3], R, t)
    c = cam_dir.name[-1]
    return {f"shape_smplx_cam{c}": smpl["shapeParameters"], f"pose_smplx_aa_cam{c}": pose_aa,
            f"trans_smplx_cam{c}": trans, f"detection_cam{c}": detected[:, np.newaxis]}


# --- Sequence ---------------------------------------------------------------------------------
def process_sequence(name: str, args: argparse.Namespace) -> dict | None:
    """All HDF5 datasets of one TotalCapture sequence (e.g. name = 's1_acting1')."""
    subject, motion = name.split("_")
    imu_base = args.tc_imu / f"{subject.upper()}_imu" / subject / name
    REF_R_S, S_a = read_xsens(Path(f"{imu_base}_Xsens.sensors"))
    W_R_REF = np.stack([W_R_TCWORLD @ R for R in read_calibration(Path(f"{imu_base}_calib_imu_ref.txt"))])
    with open(args.tc_gt / f"{name}.pkl", "rb") as f:
        pose_np = pickle.load(f, encoding="latin1")["gt"]
    vertex_ids = config.TC_SENSOR_VERTICES[subject]

    # GT mounting from the first frame of the (unsmoothed) GT.
    _, W_R_V0 = smpl_vertex_rotations(torch.from_numpy(pose_np[:1]).float().view(1, 24, 3))
    V_R_S = gt_mounting(W_R_REF, REF_R_S[0], W_R_V0[0, vertex_ids].numpy())

    trans = read_translation(args.tc_pos / subject.upper() / motion / "gt_skel_gbl_pos.txt")
    n = min(len(REF_R_S), len(pose_np), len(trans))
    pose_np, trans_np = smooth_motion(pose_np[:n].reshape(n, 24, 3).astype(np.float32), trans[:n].numpy())
    pose, trans = torch.from_numpy(pose_np).float(), torch.from_numpy(trans_np).float()
    ref_a = np.einsum("tsij,tsj->tsi", REF_R_S[:n], S_a[:n])    # measured acceleration in REF (with gravity)

    video = {}
    video_dir = args.video / subject.upper() / motion
    for cam_dir in sorted(video_dir.iterdir()) if video_dir.exists() else []:
        if (cam_dir / "subject-1.smpl").exists():
            video.update(load_video(cam_dir, n, trans_np) or {})
    if not video:
        return None

    verts, W_R_V = smpl_vertex_rotations(pose, trans)
    vrot_aa_glb = rotation_matrix_to_axis_angle(W_R_V[:, config.VERTEX_MASK].reshape(-1, 3, 3)).view(n, -1, 3)
    return {
        "motion_name": f"{subject.upper()}/{motion}",
        "smpl_pose_aa": pose_np, "smpl_tran": trans_np,
        "vrot_aa_glb": vrot_aa_glb.numpy(),
        "vacc_glb": synthesize_acc(verts[:, config.VERTEX_MASK]).numpy(),
        "ref_R_S": REF_R_S[:n].astype(np.float32), "ref_a": ref_a.astype(np.float32),
        "W_R_ref": W_R_REF.astype(np.float32), "B_R_S_aligned": V_R_S.astype(np.float32),
        "imu_vertex_ids": np.array(vertex_ids, dtype=np.int64),
        **video,
    }


def write_group(hf: h5py.File, group_name: str, sample: dict) -> None:
    grp = hf.create_group(group_name)
    grp.attrs["motion_name"] = sample.pop("motion_name")
    for key, data in sample.items():
        chunks = (200,) + data.shape[1:] if data.ndim in (2, 3) and data.shape[0] > 200 else None
        grp.create_dataset(key, data=data, chunks=chunks)


def preprocess_tc(args: argparse.Namespace) -> None:
    names = sorted(p.stem for p in args.tc_gt.glob("*.pkl"))
    with h5py.File(args.out, "w-") as hf:
        for i, name in enumerate(tqdm(names, desc="TotalCapture")):
            sample = process_sequence(name, args)
            if sample is None:
                print(f"No video estimate for {name}, skipped.")
                continue
            write_group(hf, f"sample_{i:05d}", sample)


# --- DS_VAIC ----------------------------------------------------------------------------------
def gt_vertex_imus(pose_aa: np.ndarray, trans: np.ndarray, betas: np.ndarray, gender: str,
                   batch_size: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """GT virtual IMUs of the candidate vertices: orientations W_R_V as axis-angle (T, 372, 3) and
    accelerations (T, 372, 3), from SMPL (participant shape and gender) forward kinematics."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = smpl_model(gender).to(device)
    pose = torch.tensor(pose_aa.reshape(len(pose_aa), 72), dtype=torch.float32, device=device)
    trans = torch.tensor(trans, dtype=torch.float32, device=device)
    betas = torch.tensor(betas, dtype=torch.float32, device=device).unsqueeze(0)
    verts, W_R_V = [], []
    for s in range(0, len(pose), batch_size):
        e = min(s + batch_size, len(pose))
        with torch.no_grad():
            out = model(global_orient=pose[s:e, :3], body_pose=pose[s:e, 3:], transl=trans[s:e],
                        betas=betas.expand(e - s, -1), compute_rotations=True)
        verts.append(out.vertices[:, config.VERTEX_MASK])
        W_R_V.append(out.vert_rot[:, config.VERTEX_MASK])
    verts, W_R_V = torch.cat(verts), torch.cat(W_R_V)
    vrot_aa_glb = rotation_matrix_to_axis_angle_batched(W_R_V.reshape(-1, 3, 3)).reshape(len(pose), -1, 3)
    return vrot_aa_glb.cpu().numpy(), synthesize_acc(verts).cpu().numpy()


def preprocess_vaic(args: argparse.Namespace) -> None:
    """Copy every recording of the downloaded DS_VAIC file and add the GT virtual vertex IMUs."""
    with h5py.File(args.vaic_raw, "r") as raw, h5py.File(args.out, "w-") as hf:
        for name in tqdm(list(raw.keys()), desc="DS_VAIC"):
            src = raw[name]
            grp = hf.create_group(name)
            grp.attrs["motion_name"] = src.attrs["motion_name"]
            for key in src:
                grp.create_dataset(key, data=src[key][()], dtype=src[key].dtype, chunks=src[key].chunks)
            vrot_aa_glb, vacc_glb = gt_vertex_imus(src["smpl_pose_aa"][()], src["smpl_tran"][()],
                                                   src["smpl_betas"][()], src.attrs["gender"])
            for key, data in [("vrot_aa_glb", vrot_aa_glb), ("vacc_glb", vacc_glb)]:
                grp.create_dataset(key, data=data, chunks=(200,) + data.shape[1:])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    data = config.ROOT / "data"     # where data_preprocessing/download_*.sh put the inputs
    parser.add_argument("--ds", choices=["TC", "VAIC"], default="TC")
    parser.add_argument("--tc_imu", type=Path, default=data / "TotalCapture_IMU")
    parser.add_argument("--tc_gt", type=Path, default=data / "TotalCapture_Real_60FPS")
    parser.add_argument("--tc_pos", type=Path, default=data / "TotalCapture")
    parser.add_argument("--video", type=Path, default=data / "tc_video")
    parser.add_argument("--vaic_raw", type=Path, default=config.DSVAIC_RAW_H5)
    parser.add_argument("--out", type=Path, help="default: config.DATASET_PATHS[--ds]")
    args = parser.parse_args()
    args.out = args.out or config.DATASET_PATHS[args.ds]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.ds == "TC":
        preprocess_tc(args)
    else:
        preprocess_vaic(args)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
