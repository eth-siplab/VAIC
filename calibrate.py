"""Calibrate body-worn IMUs from a video-estimated SMPL-X motion (VAIC) or with the T-Pose baseline.

    python calibrate.py --ds TC   [--fixed] [--seq_len 600] [--gt_smpl]
    python calibrate.py --ds VAIC --fixed --seq_len 1200
    python calibrate.py --ds TC --method tpose

--fixed      IMU positions are known (no localisation, Tab. 3 / Tab. 7).
--gt_smpl    use the mocap SMPL instead of the video SMPL-X ("VAIC w/ SMPL GT").
--seq_len    calibration window length in frames at 60 fps (600 / 1200 / 1800 = 10 / 20 / 30 s).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import torch
from tqdm import tqdm

import config
from vaic.calibrate import tpose, vaic
from vaic.data import CalibrationWindows, is_excluded, is_valid, prepare_inputs
from vaic.results import save_result


def run_vaic(args: argparse.Namespace, out_file: Path) -> None:
    windows = CalibrationWindows(args.data_path, args.seq_len)
    n_calibrated = 0
    for idx in tqdm(range(len(windows)), desc="VAIC"):
        window = windows[idx]
        if is_excluded(window) or not is_valid(window):
            continue
        result = vaic(prepare_inputs(window, args.gt_smpl), args.fixed)
        save_result(out_file, window["seq_name"], window["motion_name"], window["camera_index"],
                    window["start_index"], window["end_index"], result["W_R_REF"], result["V_R_S"],
                    config.VERTEX_MASK[result["vertex_idx"]].numpy(), result["cost_matrix"])
        n_calibrated += 1
    print(f"Calibrated {n_calibrated} of {len(windows)} windows.")


def run_tpose(args: argparse.Namespace, out_file: Path) -> None:
    start, end = config.TPOSE_FRAMES
    with h5py.File(args.data_path, "r") as f:
        for seq_name in tqdm(list(f.keys()), desc="T-Pose"):
            grp = f[seq_name]
            motion_name = grp.attrs["motion_name"]
            if motion_name in config.IGNORED_TC_SEQUENCES:
                continue
            result = tpose(torch.from_numpy(grp["ref_R_S"][start:end]))
            save_result(out_file, seq_name, motion_name, None, 0, grp["ref_R_S"].shape[0],
                        result["W_R_REF"], result["V_R_S"], grp["imu_vertex_ids"][:])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ds", choices=["TC", "VAIC"], required=True)
    parser.add_argument("--method", choices=["vaic", "tpose"], default="vaic")
    parser.add_argument("--fixed", action="store_true", help="IMU positions are known.")
    parser.add_argument("--gt_smpl", action="store_true", help="Use mocap SMPL instead of video SMPL-X.")
    parser.add_argument("--seq_len", type=int, default=600, help="Window length in frames (60 fps).")
    parser.add_argument("--data_path", type=Path, default=None, help="Override the dataset HDF5.")
    parser.add_argument("--out_dir", type=Path, default=None, help="Override the output folder.")
    args = parser.parse_args()
    if args.method == "tpose" and args.ds != "TC":
        raise ValueError("The T-Pose baseline is only available on TotalCapture.")

    args.data_path = args.data_path or config.DATASET_PATHS[args.ds]
    args.out_dir = args.out_dir or config.RESULTS_DIR / f"{args.ds}_{args.method}_{time.strftime('%Y%m%d-%H%M%S')}"
    args.out_dir.mkdir(parents=True, exist_ok=False)
    (args.out_dir / "args.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=1))

    out_file = args.out_dir / "calibration_results.hdf5"
    if args.method == "tpose":
        run_tpose(args, out_file)
    else:
        run_vaic(args, out_file)
    print(f"Results written to {args.out_dir}. Evaluate with: python evaluate.py --results {args.out_dir}")


if __name__ == "__main__":
    main()
