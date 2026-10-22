#!/usr/bin/env bash
#
# Download the SMPL-X v1.1 body model (neutral). Used to synthesise the virtual IMUs of the
# SMPL-X motion estimated from video.
# Requires a free account at https://smpl-x.is.tue.mpg.de.
#
# Result: data/body_models/smplx/SMPLX_NEUTRAL.npz   (dir via VAIC_BODY_MODELS)
#
# Usage: ./data_preprocessing/download_smplx.sh
#        (or sourced by download_all.sh, which calls smplx_run)
#
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

SMPLX_MD5="6eed2e6dfee62a3f4fb98dcd41f94e06"   # SMPLX_NEUTRAL.npz used for the paper

smplx_target() { echo "${VAIC_BODY_MODELS:-$DATA_DIR/body_models}/smplx/SMPLX_NEUTRAL.npz"; }

smplx_present()  { [ -f "$(smplx_target)" ]; }
smplx_accounts() { smplx_present || echo smplx; }

smplx_run() {
    local TARGET ZIP TMP NEUTRAL
    TARGET="$(smplx_target)"
    if [ -f "$TARGET" ]; then
        echo "SMPL-X model already present: $TARGET"
        md5_check "$TARGET" "$SMPLX_MD5"
        return 0
    fi

    ZIP="$DOWNLOAD_CACHE/models_smplx_v1_1.zip"
    tue_download smplx 'https://download.is.tue.mpg.de/download.php?domain=smplx&sfile=models_smplx_v1_1.zip' "$ZIP"

    TMP="$(mktemp -d)"
    trap 'rm -rf "$TMP"' RETURN
    unzip -o "$ZIP" -d "$TMP" >/dev/null

    NEUTRAL="$(find "$TMP" -name 'SMPLX_NEUTRAL.npz' -print -quit)"
    [ -n "$NEUTRAL" ] || { echo "ERROR: SMPLX_NEUTRAL.npz not found inside $ZIP" >&2; return 1; }

    mkdir -p "$(dirname "$TARGET")"
    cp "$NEUTRAL" "$TARGET"
    md5_check "$TARGET" "$SMPLX_MD5"
    echo "Installed SMPL-X model: $TARGET"
    echo "(You can delete the cached archive $ZIP to reclaim space.)"
}

# Run standalone (skipped when sourced by download_all.sh).
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    if smplx_present; then smplx_run; exit 0; fi
    if ! ask_yes_no "Download the SMPL-X v1.1 body model (login required)?"; then
        skip_note "the SMPL-X body model" "set VAIC_BODY_MODELS to the folder containing smplx/SMPLX_NEUTRAL.npz" "$(smplx_target)"
        exit 0
    fi
    smplx_run
fi
