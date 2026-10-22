#!/usr/bin/env bash
#
# Download the SMPL v1.0.0 body models (male, female). Used for the GT motion of TotalCapture (male)
# and DS_VAIC (participant gender), its virtual IMUs, and the bone-accuracy metric (male).
# Requires a free account at https://smpl.is.tue.mpg.de.
#
# Result: data/body_models/smpl/SMPL_{MALE,FEMALE}.pkl   (config.SMPL_FILES; dir via VAIC_BODY_MODELS)
#
# Usage: ./data_preprocessing/download_smpl.sh
#        (or sourced by download_all.sh, which calls smpl_run)
#
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

SMPL_MD5="a4ba3ca481e1a2b246b4202873d8addf"          # basicmodel_m_lbs_10_207_0_v1.0.0.pkl used for the paper
SMPL_FEMALE_MD5="0842c0f9b87d95730fb0e5048f05d611"   # SMPL_FEMALE.pkl used for the paper (DS_VAIC)

smpl_target()        { echo "${VAIC_BODY_MODELS:-$DATA_DIR/body_models}/smpl/SMPL_MALE.pkl"; }
smpl_female_target() { echo "${VAIC_BODY_MODELS:-$DATA_DIR/body_models}/smpl/SMPL_FEMALE.pkl"; }

smpl_present()  { [ -f "$(smpl_target)" ] && [ -f "$(smpl_female_target)" ]; }
smpl_accounts() { smpl_present || echo smpl; }

smpl_run() {
    local TARGET FTARGET ZIP TMP MALE FEMALE
    TARGET="$(smpl_target)"
    FTARGET="$(smpl_female_target)"
    if smpl_present; then
        echo "SMPL models already present: $TARGET, $FTARGET"
        md5_check "$TARGET" "$SMPL_MD5"
        md5_check "$FTARGET" "$SMPL_FEMALE_MD5"
        return 0
    fi

    ZIP="$DOWNLOAD_CACHE/SMPL_python_v.1.0.0.zip"
    tue_download smpl 'https://download.is.tue.mpg.de/download.php?domain=smpl&sfile=SMPL_python_v.1.0.0.zip' "$ZIP"

    TMP="$(mktemp -d)"
    trap 'rm -rf "$TMP"' RETURN
    unzip -o "$ZIP" -d "$TMP" >/dev/null

    MALE="$(find "$TMP" -iname 'basicmodel_m_lbs_10_207_0_v1.0.0.pkl' -print -quit)"
    [ -n "$MALE" ] || { echo "ERROR: male SMPL model not found inside $ZIP" >&2; return 1; }
    FEMALE="$(find "$TMP" -iname 'basicmodel_f_lbs_10_207_0_v1.0.0.pkl' -print -quit)"
    [ -n "$FEMALE" ] || { echo "ERROR: female SMPL model not found inside $ZIP" >&2; return 1; }

    mkdir -p "$(dirname "$TARGET")"
    cp "$MALE" "$TARGET"
    cp "$FEMALE" "$FTARGET"
    md5_check "$TARGET" "$SMPL_MD5"
    md5_check "$FTARGET" "$SMPL_FEMALE_MD5"
    echo "Installed SMPL models: $TARGET, $FTARGET"
    echo "(You can delete the cached archive $ZIP to reclaim space.)"
}

# Run standalone (skipped when sourced by download_all.sh).
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    if smpl_present; then smpl_run; exit 0; fi
    if ! ask_yes_no "Download the SMPL v1.0.0 body models (login required)?"; then
        skip_note "the SMPL body models" "set VAIC_BODY_MODELS to the folder containing smpl/SMPL_{MALE,FEMALE}.pkl" "$(dirname "$(smpl_target)")"
        exit 0
    fi
    smpl_run
fi
