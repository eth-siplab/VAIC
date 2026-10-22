"""Evaluate a calibration result folder with the five metrics of the paper.

Calibrations are evaluated on the whole recorded sequence (TotalCapture) or on the separate
validation block recorded after the calibration motion (DS_VAIC).

    python evaluate.py --results results/TC_vaic_<timestamp> [--data_path <h5>]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import torch

import config
from vaic.data import find_event_frame, masked_vertex_index, read_events
from vaic.metrics import METRICS, imu_metrics, summarize
from vaic.results import load_results
from vaic.rotations import axis_angle_to_rotation_matrix


def evaluation_range(grp: h5py.Group) -> slice:
    """Whole sequence, or the validation block if the sequence has protocol events (DS_VAIC)."""
    events = read_events(grp)
    if events is None:
        return slice(None)
    start = find_event_frame(events, "VALIDATION_DYNAMIC_START")
    end = max(int(round(e["t"] * config.FPS)) for e in events if "VALIDATION" in " ".join(e["label"].split()))
    return slice(start, end + 1)


def evaluate_window(grp: h5py.Group, result: dict) -> list[dict]:
    """Per-IMU metrics of one calibrated window."""
    frames = evaluation_range(grp)
    REF_R_S = torch.from_numpy(grp["ref_R_S"][frames])
    vrot_aa_glb = torch.from_numpy(grp["vrot_aa_glb"][frames])
    gt_vertex_ids = grp["imu_vertex_ids"][:]
    rows = []
    for i, s in enumerate(result["sensor_indices"]):
        W_R_Vt_smpl = axis_angle_to_rotation_matrix(vrot_aa_glb[:, masked_vertex_index([gt_vertex_ids[s]])[0]])
        row = imu_metrics(result["W_R_REF"][i], result["V_R_S"][i], int(result["vertex_ids"][i]),
                          torch.from_numpy(grp["W_R_ref"][s]), torch.from_numpy(grp["B_R_S_aligned"][s]),
                          int(gt_vertex_ids[s]), REF_R_S[:, s], W_R_Vt_smpl)
        rows.append({"seq_name": result["seq_name"], "motion_name": result["motion_name"],
                     "camera_index": result["camera_index"], "start_index": result["start_index"],
                     "sensor_index": int(s), **row})
    return rows


def print_summary(summary: dict, n_windows: int, n_rows: int) -> None:
    print(f"{n_windows} calibration windows, {n_rows} IMUs")
    print(f"{'Bone Acc. (%)':>14} {'Pos. Err. (cm)':>15} {'AE W_R_V (deg)':>15} {'AE V_R_S (deg)':>15} "
          f"{'AE W_R_REF (deg)':>17}")
    print(f"{summary['bone_acc'][0]:>14.1f} "
          + " ".join(f"{summary[m][0]:>8.1f} ± {summary[m][1]:<4.1f}" for m in METRICS[1:]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, required=True, help="Folder written by calibrate.py.")
    parser.add_argument("--data_path", type=Path, default=None, help="Dataset HDF5 (default: from args.json).")
    args = parser.parse_args()

    data_path = args.data_path or Path(json.loads((args.results / "args.json").read_text())["data_path"])
    windows = load_results(args.results / "calibration_results.hdf5")
    rows = []
    with h5py.File(data_path, "r") as f:
        for result in windows:
            rows.extend(evaluate_window(f[result["seq_name"]], result))

    summary = summarize(rows)
    print_summary(summary, len(windows), len(rows))
    (args.results / "metrics.json").write_text(json.dumps({"summary": summary, "per_imu": rows}, indent=1))


if __name__ == "__main__":
    main()
