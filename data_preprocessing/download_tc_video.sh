#!/usr/bin/env bash
#
# Download our SMPL-X estimates of the TotalCapture videos (one folder per sequence and
# camera) from Google Drive. No login required.
#
# Result: data/tc_video/S{1..5}/<motion>/TC_S{1..5}_<motion>_cam<c>/{subject-1.smpl, results_detected.pkl}
#         (preprocess.py --video)
#
# Usage: ./data_preprocessing/download_tc_video.sh
#        (or sourced by download_all.sh, which calls tc_video_run)
# Env:   TC_VIDEO_DIR  target dir (default data/tc_video)
#
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

# Google Drive file ID of tc_video.zip (the part after /file/d/ in the share link).
GDRIVE_ID="16Ke3VoH71E8f3h1MXTGWyYHY4VJQZDAc"

tc_video_dir() { echo "${TC_VIDEO_DIR:-$DATA_DIR/tc_video}"; }

tc_video_present() {
    local d; d="$(tc_video_dir)"
    [ -n "$(find "$d" -name 'subject-1.smpl' -print -quit 2>/dev/null)" ]
}

# Google Drive download — no login required.
tc_video_accounts() { :; }

tc_video_run() {
    local DST STAGE archive root
    DST="$(tc_video_dir)"
    if tc_video_present; then echo "TotalCapture video estimates already present in $DST"; return 0; fi

    STAGE="$(mktemp -d)"
    trap 'rm -rf "$STAGE"' RETURN
    ( cd "$STAGE" && gdown "https://drive.google.com/uc?id=${GDRIVE_ID}" )
    archive="$(find "$STAGE" -maxdepth 1 -type f -name '*.zip' -print -quit)"
    [ -n "$archive" ] || { echo "ERROR: download produced no zip file" >&2; return 1; }
    unzip -q "$archive" -d "$STAGE/extracted" -x "__MACOSX/*"

    # The archive root is the folder that contains S1..S5.
    root="$(dirname "$(find "$STAGE/extracted" -maxdepth 3 -type d -name S1 -print -quit)")"
    [ -d "$root/S1" ] || { echo "ERROR: no S1 folder found inside $(basename "$archive")" >&2; return 1; }
    mkdir -p "$DST"
    mv "$root"/S* "$DST"/
    echo "TotalCapture video estimates installed in $DST ($(find "$DST" -name 'subject-1.smpl' | wc -l) camera views)"
}

# Run standalone (skipped when sourced by download_all.sh).
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    if tc_video_present; then tc_video_run; exit 0; fi
    if ! ask_yes_no "Download our SMPL-X estimates of the TotalCapture videos (~0.4 GB, Google Drive)?"; then
        skip_note "the TotalCapture video estimates" "pass them to preprocess.py --video" "$(tc_video_dir)"
        exit 0
    fi
    tc_video_run
fi
