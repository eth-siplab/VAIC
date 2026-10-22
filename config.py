"""Paths, dataset constants and the hyperparameters used in the paper."""
from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
ASSETS_DIR = ROOT / "assets"

# --- Paths (override with environment variables) ---------------------------------------
# Directory with `smpl/SMPL_{MALE,FEMALE}.pkl` and `smplx/SMPLX_NEUTRAL.npz` (see README).
BODY_MODEL_DIR = Path(os.getenv("VAIC_BODY_MODELS", ROOT / "data" / "body_models"))
SMPL_FILES = {g: BODY_MODEL_DIR / "smpl" / f"SMPL_{g.upper()}.pkl" for g in ("male", "female")}
SMPL_MALE_FILE = SMPL_FILES["male"]
DATASET_PATHS = {
    "TC": Path(os.getenv("VAIC_TC_H5", ROOT / "data" / "tc.h5")),
    "VAIC": Path(os.getenv("VAIC_DSVAIC_H5", ROOT / "data" / "ds_vaic.h5")),
}
DSVAIC_RAW_H5 = ROOT / "data" / "ds_vaic_raw.h5"     # the downloaded DS_VAIC file, input of preprocess.py --ds VAIC
RESULTS_DIR = Path(os.getenv("VAIC_RESULTS", ROOT / "results"))

# --- Candidate vertices -------------------------------------------------------------------
# 372 SMPL vertices: a uniform subsample of the body surface (face, hands, feet and crotch
# removed) plus a set of always-included common mounting locations.
VERTEX_MASK = torch.from_numpy(np.load(ASSETS_DIR / "vertex_mask.npy"))
# Rotation from the surface tangent frame T ("SCAN") to the vertex frame V, per SMPL vertex.
V_R_T = torch.from_numpy(np.load(ASSETS_DIR / "sensor_canonical_frames.npz")["v_R_scan"])
# SMPL vertex index -> closest SMPL-X vertex index.
with open(ASSETS_DIR / "smpl_to_smplx_vertex_map.pkl", "rb") as f:
    SMPL_TO_SMPLX_VERTEX_MAP: dict = pickle.load(f)

# --- Datasets -----------------------------------------------------------------------------
FPS = 60.0
GRAVITY_Y_UP = torch.tensor([0.0, 9.81, 0.0])

TC_SENSOR_NAMES = ["Head", "Sternum", "Pelvis",
                   "L_UpArm", "R_UpArm", "L_LowArm", "R_LowArm",
                   "L_UpLeg", "R_UpLeg", "L_LowLeg", "R_LowLeg",
                   "L_Foot", "R_Foot"]
# Annotated SMPL vertex of every TotalCapture sensor, per subject (order as TC_SENSOR_NAMES).
TC_SENSOR_VERTICES = {
    "s1": [411, 4184, 3021, 1389, 5365, 1961, 5424, 848, 4358, 1176, 4598, 3334, 6799],
    "s2": [411, 4184, 3021, 1392, 4916, 1592, 5424, 847, 4391, 1371, 4588, 3367, 6768],
    "s3": [3761, 4184, 3021, 1392, 6438, 1962, 5423, 848, 4358, 1176, 4603, 3399, 6770],
    "s4": [411, 4184, 3097, 1444, 4918, 1962, 5430, 848, 4333, 1371, 4846, 3368, 6768],
    "s5": [412, 1255, 3097, 1680, 5167, 1969, 5705, 847, 4336, 1175, 4663, 3469, 6770],
}

# Calibration windows that are excluded from the evaluation.
IGNORED_TC_SEQUENCES = ["S5/freestyle3"]                      # IMU/GT time-sync issues
BROKEN_TC_CAMERAS = [3, 8]                                     # persistent blur / detection failures
GT_ISSUES = [("S2/freestyle3", 1210), ("S4/acting3", 1210)]    # (motion, start frame): GT hand flips

# --- Hyperparameters (paper, Sec. 4 and App. B.1) -------------------------------------------
LAMBDA_ACC = 5e-6
LAMBDA_NORMAL = 0.1
MAX_ITER = 50
CUT_OFF_FRONT = 10          # frames skipped at the start of every calibration window
TPOSE_FRAMES = (10, 30)     # frames of the initial T-pose used by the T-Pose baseline
