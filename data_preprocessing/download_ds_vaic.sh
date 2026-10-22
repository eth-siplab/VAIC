#!/usr/bin/env bash
#
# Download DS_VAIC (one HDF5: GT SMPL motion + body shape, raw IMU data, GT calibration, video
# SMPL-X estimates and the protocol segments of every recording) from Google Drive. No login required.
# Then run `python preprocess.py --ds VAIC` to add the GT virtual vertex IMUs (-> data/ds_vaic.h5).
#
# Result: data/ds_vaic_raw.h5  (preprocess.py --vaic_raw)
#
# Usage: ./data_preprocessing/download_ds_vaic.sh
#        (or sourced by download_all.sh, which calls ds_vaic_run)
# Env:   DS_VAIC_RAW  target file (default data/ds_vaic_raw.h5)
#
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

# Google Drive file ID of ds_vaic.zip (the part after /file/d/ in the share link).
DS_VAIC_GDRIVE_ID="1o7L7IrIc0oR0lGkWIO2bEnz-7_JtGWlm"
DS_VAIC_MD5="cd38e143d5c9f5c8df69c166c1a142e4"     # ds_vaic_raw.h5

ds_vaic_file() { echo "${DS_VAIC_RAW:-$DATA_DIR/ds_vaic_raw.h5}"; }

ds_vaic_present() { [ -f "$(ds_vaic_file)" ]; }

# Google Drive download — no login required.
ds_vaic_accounts() { :; }

ds_vaic_run() {
    local DST STAGE archive h5
    DST="$(ds_vaic_file)"
    if ds_vaic_present; then echo "DS_VAIC already present at $DST"; return 0; fi

    STAGE="$(mktemp -d)"
    trap 'rm -rf "$STAGE"' RETURN
    ( cd "$STAGE" && gdown "https://drive.google.com/uc?id=${DS_VAIC_GDRIVE_ID}" )
    archive="$(find "$STAGE" -maxdepth 1 -type f -name '*.zip' -print -quit)"
    [ -n "$archive" ] || { echo "ERROR: download produced no zip file" >&2; return 1; }
    unzip -q "$archive" -d "$STAGE/extracted" -x "__MACOSX/*"
    h5="$(find "$STAGE/extracted" -name 'ds_vaic_raw.h5' -print -quit)"
    [ -n "$h5" ] || { echo "ERROR: no ds_vaic_raw.h5 inside $(basename "$archive")" >&2; return 1; }
    md5_check "$h5" "$DS_VAIC_MD5"
    mkdir -p "$(dirname "$DST")"
    mv "$h5" "$DST"
    echo "DS_VAIC installed at $DST (next: python preprocess.py --ds VAIC)"
}

# Run standalone (skipped when sourced by download_all.sh).
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    if ds_vaic_present; then ds_vaic_run; exit 0; fi
    if ! ask_yes_no "Download DS_VAIC (~0.2 GB, Google Drive)?"; then
        skip_note "DS_VAIC" "pass it to preprocess.py --ds VAIC --vaic_raw" "$(ds_vaic_file)"
        exit 0
    fi
    ds_vaic_run
fi
