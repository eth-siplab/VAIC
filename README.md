<div align="center">

<h1>Vision-Aided IMU Calibrator: Robust On-body IMU Calibration from a Single Video (SIGGRAPH Asia 2026)</h1>

<p>
	<a href="https://dominik-hollidt.github.io/">Dominik Hollidt</a>, <a href="https://www.christianholz.net">Christian Holz</a><br>
	<a href="https://siplab.org">Sensing, Interaction & Perception Lab</a>, Department of Computer Science, ETH Zürich, Switzerland<br>
	<a href="https://siplab.org/projects/VAIC">Project Page</a> · <a href="https://doi.org/10.1145/3829340.3842338">Paper</a>
</p>

</div>

## Abstract

Inertial motion capture depends on expressing each IMU's orientations and accelerations in a shared world frame and
knowing how each sensor is rotated relative to the body segment it is attached to. However, in practice, this
requires careful placement and manual calibration routines (T-pose) that do not scale to arbitrary sensor sets or
in-the-wild use. We present VAIC, a visual–inertial calibration method that uses a few seconds of monocular video of
arbitrary user motion plus uncalibrated IMU streams to jointly recover (1) each sensor's body location (as a vertex on
an SMPL mesh), (2) its sensor-to-body mounting rotation, and (3) the global reference alignment to the world frame.
Our method operates without predefined poses, fixed sensor configurations, or pre-calibrated hardware. Instead, we
treat calibration as an optimization problem: from video-based SMPL recovery, VAIC searches for the sensor-to-vertex
alignment that best explains the measured IMU signals given these vertex kinematics.

## Code overview

```
preprocess.py        builds the HDF5 files: TotalCapture (original data + our video estimates) and DS_VAIC
calibrate.py         calibration (VAIC or the T-Pose baseline) on TotalCapture or DS_VAIC
evaluate.py          the five metrics of the paper for a calibration result folder
vaic/                virtual vertex IMUs (body.py), LM solver (optim.py), assignment + W_R_REF (calibrate.py)
assets/              candidate vertex set, surface tangent frame of every SMPL vertex, SMPL→SMPL-X vertex map
data_preprocessing/  download scripts for body models and data
```

Frames: `W` world, `REF` the reference frame each IMU reports in, `S` sensor, `V` the SMPL vertex the IMU is mounted
on. `A_R_B` rotates from frame `B` to frame `A`. VAIC estimates, per IMU, the vertex, `V_R_S` and `W_R_REF`.

## 1. Install

Creates the conda environment `VAIC` (Python 3.8.10, PyTorch 2.3.1 / CUDA 12.1) with the pinned dependencies,
including our [smplx fork](https://github.com/eth-siplab/smplx) that adds per-vertex rotations
(`compute_rotations=True` → `vert_rot`).

```bash
./install.sh            # ./install.sh --name myenv for another env name
conda activate VAIC
```

## 2. Download body models and data

```bash
./data_preprocessing/download_all.sh
# or individually: download_smpl.sh  download_smplx.sh  download_tc.sh  download_tc_video.sh  download_ds_vaic.sh
```

All logins are asked for up front, then everything downloads unattended. Accounts needed (free registration):

| Account | For | Lands at |
|---|---|---|
| [SMPL](https://smpl.is.tue.mpg.de) | SMPL v1.0.0 male and female models | `data/body_models/smpl/SMPL_{MALE,FEMALE}.pkl` |
| [SMPL-X](https://smpl-x.is.tue.mpg.de) | SMPL-X v1.1 neutral model | `data/body_models/smplx/SMPLX_NEUTRAL.npz` |
| [TotalCapture (cvssp)](https://cvssp.org/data/totalcapture/) | Vicon skeleton + raw Xsens IMU data | `data/TotalCapture/`, `data/TotalCapture_IMU/` |
| [DIP-IMU](https://dip.is.tue.mpg.de) | TotalCapture SMPL ground truth (`TotalCapture_Real_60FPS`) | `data/TotalCapture_Real_60FPS/` |
| none ([Google Drive](https://drive.google.com/drive/folders/1fHKJ2mUvK5Hfa8a3Cgs8BN7YPJUGwhtb)) | our SMPL-X estimates of the TotalCapture videos | `data/tc_video/` |
| none ([Google Drive](https://drive.google.com/drive/folders/1fHKJ2mUvK5Hfa8a3Cgs8BN7YPJUGwhtb)) | DS_VAIC | `data/ds_vaic_raw.h5` |

Body models in another folder: `export VAIC_BODY_MODELS=<dir containing smpl/ and smplx/>`.

## 3. Preprocess

```bash
python preprocess.py              # TotalCapture -> data/tc.h5 (inputs default to the download locations above)
python preprocess.py --ds VAIC    # DS_VAIC      -> data/ds_vaic.h5
```

Another output location: `--out <file>` plus `export VAIC_TC_H5=<file>` / `VAIC_DSVAIC_H5=<file>`. For TotalCapture,
sequence `S5/freestyle3`, cameras 3 and 8, and the windows listed in `config.py` are excluded from the evaluation
because of known data issues.

**DS_VAIC.** `ds_vaic_raw.h5` has one group per recording (8 participants, pseudonymized as
`P<NN>_<gender>/take<k>`; 21 recordings of about 3.5 min at 60 fps with a video estimate; 6 Xsens IMUs):

| Key | Content |
|---|---|
| `smpl_pose_aa`, `smpl_tran`, `smpl_betas`, attribute `gender` | GT SMPL motion (optical mocap) and body shape, world frame W |
| `ref_R_S`, `ref_a` | raw IMU orientations and accelerations in each IMU's reference frame REF |
| `W_R_ref`, `B_R_S_aligned`, `imu_vertex_ids` | GT calibration: W_R_REF, V_R_S and the SMPL vertex of every IMU |
| `pose_smplx_aa_cam0`, `trans_smplx_cam0`, `shape_smplx_cam0`, `detection_cam0` | SMPL-X estimated from the video |
| `events_json` | protocol segments, time stamps in s: poses, the calibration motion `DYNAMIC_START/END`, the validation block `VALIDATION_*` |

`preprocess.py --ds VAIC` adds the GT virtual vertex IMUs `vrot_aa_glb`, `vacc_glb` of all candidate vertices. On a
GPU this reproduces the paper data exactly. On the CPU, vertex orientations close to 180° differ slightly (the
axis-angle conversion is ill-conditioned there); the Tab. 3 values do not change.

## 4. Calibrate and evaluate

```bash
python calibrate.py --ds TC --seq_len 600                          # -> results/TC_vaic_<timestamp>/
python evaluate.py  --results results/TC_vaic_<timestamp>
```

`--fixed`: IMU positions are known (only mounting and world alignment are estimated). `--gt_smpl`: mocap SMPL instead
of the video estimate. `--method tpose`: T-Pose baseline. `--seq_len`: calibration window in frames at 60 fps.

Expected results (mean ± sample standard deviation over all calibrated (window, IMU) pairs; TotalCapture is evaluated
on the whole sequence, DS_VAIC on its separate validation block):

| Table | `calibrate.py` arguments | Bone Acc. (%) | Pos. Err. (cm) | AE W_R_V (°) | AE V_R_S (°) | AE W_R_REF (°) |
|---|---|---|---|---|---|---|
| 1 | `--ds TC --seq_len 600` | 95.5 | 6.0 ± 8.7 | 16.6 ± 11.6 | 16.5 ± 13.6 | 11.7 ± 9.2 |
| 1 | `--ds TC --seq_len 1200` | 95.9 | 5.8 ± 8.6 | 15.8 ± 10.8 | 15.8 ± 12.2 | 10.5 ± 7.3 |
| 1 | `--ds TC --seq_len 1800` | 95.0 | 6.2 ± 8.9 | 15.5 ± 9.2 | 15.5 ± 10.2 | 9.6 ± 5.2 |
| 1 | `--ds TC --seq_len 600 --gt_smpl` | 99.3 | 3.6 ± 3.7 | 8.0 ± 5.4 | 10.2 ± 6.8 | 5.6 ± 4.1 |
| 7 | `--ds TC --seq_len 600 --fixed` | – | – | 15.3 ± 5.8 | 14.0 ± 7.3 | 11.4 ± 6.3 |
| 7 | `--ds TC --seq_len 1200 --fixed` | – | – | 14.5 ± 5.5 | 13.7 ± 7.6 | 10.3 ± 6.3 |
| 7 | `--ds TC --seq_len 1800 --fixed` | – | – | 14.4 ± 5.5 | 13.9 ± 7.8 | 9.6 ± 6.0 |
| 7 | `--ds TC --seq_len 600 --fixed --gt_smpl` | – | – | 7.9 ± 3.7 | 10.0 ± 5.9 | 6.5 ± 5.0 |
| 7 | `--ds TC --method tpose` | – | – | 16.3 ± 9.4 | 18.5 ± 10.1 | 8.6 ± 6.5 |
| 3 | `--ds VAIC --seq_len 1200 --fixed` | – | – | 17.4 ± 8.9 | 15.8 ± 8.3 | 14.2 ± 8.0 |

All rows were computed on an RTX 4090; other GPUs can show small deviations. A TotalCapture run takes about 2 h
on one GPU; most of the time is spent synthesising the SMPL-X vertex IMUs on the CPU.

## License

The code is released under the [GPL-3.0](https://www.gnu.org/licenses/gpl-3.0.html) license (see `LICENSE`); parts
of it are adapted from [PIP](https://github.com/Xinyu-Yi/PIP) (GPL-3.0). The data (DS_VAIC and our SMPL-X estimates
of the TotalCapture videos) is released under the [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/)
license (see `LICENSE_DATA.txt`, also included in the data archives). SMPL, SMPL-X, TotalCapture and DIP-IMU are
subject to their own licenses; register on the respective websites and agree to their terms before downloading.

## Citation

```bibtex
@inproceedings{hollidt2026vaic,
  title={Vision-Aided IMU Calibrator: Robust On-body IMU Calibration from a Single Video},
  author={Hollidt, Dominik and Holz, Christian},
  booktitle={SIGGRAPH Asia 2026 Conference Papers},
  year={2026},
  doi={10.1145/3829340.3842338}
}
```
