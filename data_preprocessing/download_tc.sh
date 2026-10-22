#!/usr/bin/env bash
#
# Download the TotalCapture data that preprocess.py needs (three parts):
#   1. Vicon ground-truth global skeleton (S1..S5) from cvssp.org — the
#      "position and orientation" archives (sX_vicon_pos_ori.tar.gz), which contain the
#      gt_skel_gbl_pos.txt used for the root translation.
#        -> data/TotalCapture/S{1..5}/<motion>/gt_skel_gbl_pos.txt          (--tc_pos)
#   2. Raw Xsens IMU data (S1..S5) from cvssp.org: <seq>_Xsens.sensors and the
#      <seq>_calib_imu_ref.txt / <seq>_calib_imu_bone.txt calibration files.
#        -> data/TotalCapture_IMU/S{1..5}_imu/s{1..5}/<seq>_Xsens.sensors    (--tc_imu)
#      Parts 1 and 2 require a cvssp TotalCapture account (its own, non-MPG login).
#   3. SMPL ground truth recomputed by DIP-IMU (TotalCapture_Real_60FPS) from
#      download.is.tue.mpg.de (dip account).
#        -> data/TotalCapture_Real_60FPS/<seq>.pkl                          (--tc_gt)
#
# Usage: ./data_preprocessing/download_tc.sh
#        (or sourced by download_all.sh, which calls tc_run)
# Env:   TC_OFFICIAL_DIR, TC_IMU_DIR, TC_GT_DIR  target dirs
#        TC_IMU_URL  base URL of the IMU archives <TC_IMU_URL>/S1_imu.tar.gz .. S5_imu.tar.gz
#
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

TC_SUBJECTS=(S1 S2 S3 S4 S5)
tc_official_dir() { echo "${TC_OFFICIAL_DIR:-$DATA_DIR/TotalCapture}"; }
tc_imu_dir()      { echo "${TC_IMU_DIR:-$DATA_DIR/TotalCapture_IMU}"; }
tc_gt_dir()       { echo "${TC_GT_DIR:-$DATA_DIR/TotalCapture_Real_60FPS}"; }
TC_IMU_URL="${TC_IMU_URL:-https://cvssp.org/data/totalcapture/data/dataset/imu}"

tc_official_present() {
    local d s; d="$(tc_official_dir)"
    for s in "${TC_SUBJECTS[@]}"; do
        [ -n "$(find "$d/$s" -name 'gt_skel_gbl_pos.txt' -print -quit 2>/dev/null)" ] || return 1
    done
}
tc_imu_present() {
    local d s; d="$(tc_imu_dir)"
    for s in "${TC_SUBJECTS[@]}"; do
        [ -n "$(find "$d/${s}_imu/${s,,}" -name '*_Xsens.sensors' -print -quit 2>/dev/null)" ] || return 1
    done
}
tc_gt_present() {
    local d; d="$(tc_gt_dir)"
    [ -d "$d" ] && [ -n "$(ls -A "$d" 2>/dev/null)" ]
}

tc_present()  { tc_official_present && tc_imu_present && tc_gt_present; }
tc_accounts() {
    { tc_official_present && tc_imu_present; } || echo cvssp
    tc_gt_present || echo dip
}

# Part 1: Vicon ground-truth global skeleton S1..S5 (cvssp.org).
tc_official_run() {
    local TC_OFFICIAL s ls tgz tmp gt subjectdir
    TC_OFFICIAL="$(tc_official_dir)"
    if tc_official_present; then
        echo "TotalCapture ground-truth skeleton (S1..S5) already present at $TC_OFFICIAL"
        return 0
    fi
    cvssp_login
    for s in "${TC_SUBJECTS[@]}"; do
        ls="${s,,}"                                   # S1 -> s1 (cvssp uses lowercase here)
        tgz="$DOWNLOAD_CACHE/${ls}_vicon_pos_ori.tar.gz"
        echo ">>> ${ls}_vicon_pos_ori.tar.gz"
        cvssp_download "https://cvssp.org/data/totalcapture/data/dataset/vicon/${ls}_vicon_pos_ori.tar.gz" "$tgz"

        # Extract to a temp dir, find a gt_skel_gbl_pos.txt, and move its <subject>
        # dir (two levels up: <subject>/<motion>/gt_skel_gbl_pos.txt) into place.
        tmp="$(mktemp -d)"
        tar -xzf "$tgz" -C "$tmp"
        gt="$(find "$tmp" -name 'gt_skel_gbl_pos.txt' -print -quit)"
        [ -n "$gt" ] || { echo "ERROR: gt_skel_gbl_pos.txt not found in $tgz" >&2; rm -rf "$tmp"; return 1; }
        subjectdir="$(dirname "$(dirname "$gt")")"
        mkdir -p "$TC_OFFICIAL"
        rm -rf "${TC_OFFICIAL:?}/$s"
        mv "$subjectdir" "$TC_OFFICIAL/$s"
        rm -rf "$tmp"
    done
    echo "TotalCapture ground-truth skeleton extracted to $TC_OFFICIAL ($(ls "$TC_OFFICIAL" | tr '\n' ' '))"
}

# Part 2: raw Xsens IMU data S1..S5 (cvssp.org).
tc_imu_run() {
    local TC_IMU s ls url tgz tmp sensors srcdir dst
    TC_IMU="$(tc_imu_dir)"
    if tc_imu_present; then
        echo "TotalCapture IMU data (S1..S5) already present at $TC_IMU"
        return 0
    fi
    cvssp_login
    for s in "${TC_SUBJECTS[@]}"; do
        ls="${s,,}"
        url="$TC_IMU_URL/${s}_imu.tar.gz"
        tgz="$DOWNLOAD_CACHE/$(basename "$url")"
        echo ">>> $(basename "$url")"
        cvssp_download "$url" "$tgz"

        # Extract to a temp dir and move the folder holding the <seq>_Xsens.sensors files
        # to <TC_IMU>/S1_imu/s1/, whatever the layout inside the archive.
        tmp="$(mktemp -d)"
        tar -xzf "$tgz" -C "$tmp"
        sensors="$(find "$tmp" -name "${ls}_*_Xsens.sensors" -print -quit)"
        [ -n "$sensors" ] || { echo "ERROR: no ${ls}_*_Xsens.sensors found in $tgz" >&2; rm -rf "$tmp"; return 1; }
        srcdir="$(dirname "$sensors")"
        [ -n "$(find "$srcdir" -name "${ls}_*_calib_imu_ref.txt" -print -quit)" ] \
            || { echo "ERROR: no ${ls}_*_calib_imu_ref.txt next to the .sensors files in $tgz" >&2; rm -rf "$tmp"; return 1; }
        dst="$TC_IMU/${s}_imu/$ls"
        mkdir -p "$(dirname "$dst")"
        rm -rf "${TC_IMU:?}/${s}_imu/$ls"
        mv "$srcdir" "$dst"
        rm -rf "$tmp"
    done
    echo "TotalCapture IMU data extracted to $TC_IMU ($(ls "$TC_IMU" | tr '\n' ' '))"
}

# Part 3: DIP-recomputed SMPL GT (is.tue.mpg.de, dip account).
tc_gt_run() {
    local TC_GT ZIP tmp src
    TC_GT="$(tc_gt_dir)"
    if tc_gt_present; then
        echo "TotalCapture SMPL ground truth already present at $TC_GT"
        return 0
    fi
    ZIP="$DOWNLOAD_CACHE/TotalCapture_Real_60FPS.zip"
    tue_download dip 'https://download.is.tue.mpg.de/download.php?domain=dip&resume=1&sfile=TotalCapture_Real_60FPS.zip' "$ZIP"

    # Extract to temp, then move the *.pkl files into place (handles nested or flat zip).
    tmp="$(mktemp -d)"
    trap 'rm -rf "$tmp"' RETURN
    unzip -oq "$ZIP" -d "$tmp"
    src="$(dirname "$(find "$tmp" -name '*.pkl' -print -quit)")"
    [ -n "$src" ] || { echo "ERROR: no .pkl files found inside $ZIP" >&2; return 1; }
    mkdir -p "$TC_GT"
    mv -f "$src"/*.pkl "$TC_GT"/
    echo "TotalCapture SMPL ground truth extracted to $TC_GT ($(ls "$TC_GT" | wc -l) files)"
    echo "(You can delete the cached archives in $DOWNLOAD_CACHE to reclaim space.)"
}

# All parts (each no-ops if already present) — used by download_all.sh.
tc_run() { tc_official_run; tc_imu_run; tc_gt_run; }

# Run standalone (skipped when sourced by download_all.sh).
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    if tc_official_present; then
        tc_official_run
    elif ask_yes_no "Download the TotalCapture Vicon ground-truth skeleton (S1..S5) from cvssp.org?"; then
        tc_official_run
    else
        skip_note "the TotalCapture ground-truth skeleton (S1..S5)" "pass it to preprocess.py --tc_pos" "$(tc_official_dir)"
    fi

    if tc_imu_present; then
        tc_imu_run
    elif ask_yes_no "Download the TotalCapture raw IMU data (S1..S5) from cvssp.org?"; then
        tc_imu_run
    else
        skip_note "the TotalCapture IMU data" "pass it to preprocess.py --tc_imu" "$(tc_imu_dir)"
    fi

    if tc_gt_present; then
        tc_gt_run
    elif ask_yes_no "Download the TotalCapture SMPL ground truth (TotalCapture_Real_60FPS, dip account)?"; then
        tc_gt_run
    else
        skip_note "the TotalCapture SMPL ground truth" "pass it to preprocess.py --tc_gt" "$(tc_gt_dir)"
    fi
fi
