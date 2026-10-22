#!/usr/bin/env bash
#
# Shared helpers for the VAIC data-preparation scripts (data_preprocessing/*.sh).
# Sourced by each download_*.sh; not meant to be run on its own.
#
# Conventions:
#   * Each download_*.sh exposes three functions, so download_all.sh can source it
#     and do ALL the prompting up front (selection + logins), then download
#     unattended:
#       <name>_present   -> 0 if the target is already on disk
#       <name>_accounts  -> prints the login domains still needed (empty if none)
#       <name>_run       -> performs the download (re-checks presence, skips if done)
#     Run directly, each script still prompts-then-downloads on its own.
#   * Credentials live ONLY in non-exported shell variables of the running process
#     (never `export`ed, never written to disk) and are handed to wget on the command
#     line of a single short-lived process.
#   * Set AUTO_YES=1 to answer yes to every prompt (non-interactive use).
#   * Large archives are cached under data/downloads/ and extracted via $TMPDIR.
#
set -eo pipefail

# Idempotent: download_all.sh sources this once and then sources every
# download_*.sh (which re-source it). Do the one-time setup only on first load.
[ -n "${_VAIC_COMMON_SOURCED:-}" ] && return 0
_VAIC_COMMON_SOURCED=1

_DP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$_DP_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Activate the VAIC conda env (created by ./install.sh) so `python` and `gdown` are available.
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
VAIC_ENV="${VAIC_ENV:-VAIC}"
if [ -f "$CONDA_SH" ]; then
    # shellcheck disable=SC1090
    source "$CONDA_SH"
    conda activate "$VAIC_ENV"
fi

DATA_DIR="${DATA_DIR:-data}"
DOWNLOAD_CACHE="${DOWNLOAD_CACHE:-$DATA_DIR/downloads}"

# URL-encode a string (for the is.tue.mpg.de login form).
urle() { [[ "${1}" ]] || return 1; local LANG=C i x; for (( i = 0; i < ${#1}; i++ )); do x="${1:i:1}"; [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"; done; echo; }

# ask_yes_no "question"  -> 0 (yes) / 1 (no). AUTO_YES=1 forces yes.
ask_yes_no() {
    local q="$1" reply
    if [ "${AUTO_YES:-0}" = "1" ]; then echo "$q [auto-yes]"; return 0; fi
    read -r -p "$q [y/N] " reply
    case "${reply,,}" in y|yes) return 0 ;; *) return 1 ;; esac
}

# Per-account credential cache (in-memory only, never written to disk).
declare -p _TUE_USER_E >/dev/null 2>&1 || declare -gA _TUE_USER_E _TUE_PASS_E

# tue_login <domain>   (domain: smpl | smplx | dip)
# Each is.tue.mpg.de site needs its OWN account, EXCEPT the TotalCapture SMPL ground
# truth, which is hosted under the same dip account (so reuse domain=dip there).
# Credentials are cached per-domain for the process and url-encoded once.
# Precedence: cached this session > $TUE_<DOMAIN>_USERNAME/_PASSWORD env > prompt.
tue_login() {
    local domain="$1" UD="${1^^}" uvar pvar u p site
    [ -n "${_TUE_USER_E[$domain]:-}" ] && return 0
    site="$domain"; [ "$domain" = smplx ] && site="smpl-x"     # download domain smplx, website smpl-x
    uvar="TUE_${UD}_USERNAME"; pvar="TUE_${UD}_PASSWORD"; u="${!uvar:-}"; p="${!pvar:-}"
    if [ -n "$u" ] && [ -n "$p" ]; then
        echo "Using ${site}.is.tue.mpg.de credentials from \$$uvar/\$$pvar."
    else
        echo "A free account at https://${site}.is.tue.mpg.de is required for this download."
        read -r -p "Username (${site}.is.tue.mpg.de): " u
        read -r -s -p "Password (${site}.is.tue.mpg.de): " p; echo
    fi
    _TUE_USER_E[$domain]="$(urle "$u")"; _TUE_PASS_E[$domain]="$(urle "$p")"
}

# tue_download <domain> <url> <out> -> authenticated download (resumable).
tue_download() {
    local domain="$1" url="$2" out="$3"
    tue_login "$domain"
    mkdir -p "$(dirname "$out")"
    wget --post-data "username=${_TUE_USER_E[$domain]}&password=${_TUE_PASS_E[$domain]}" \
         "$url" -O "$out" --no-check-certificate --continue
}

# cvssp credentials (the TotalCapture data on cvssp.org — its own non-MPG account).
declare -p _CVSSP_USER >/dev/null 2>&1 || _CVSSP_USER="" _CVSSP_PASS=""
cvssp_login() {
    [ -n "$_CVSSP_USER" ] && return 0
    if [ -n "${CVSSP_USERNAME:-}" ] && [ -n "${CVSSP_PASSWORD:-}" ]; then
        _CVSSP_USER="$CVSSP_USERNAME"; _CVSSP_PASS="$CVSSP_PASSWORD"
        echo "Using cvssp credentials from \$CVSSP_USERNAME/\$CVSSP_PASSWORD."
        return 0
    fi
    echo "The TotalCapture data on cvssp.org requires its own account (not an MPG one)."
    read -r -p "Username (cvssp TotalCapture): " _CVSSP_USER
    read -r -s -p "Password (cvssp TotalCapture): " _CVSSP_PASS; echo
}

# cvssp_download <url> <out> -> HTTP-basic-auth download (resumable).
cvssp_download() {
    local url="$1" out="$2"
    cvssp_login
    mkdir -p "$(dirname "$out")"
    wget --user="$_CVSSP_USER" --password="$_CVSSP_PASS" --auth-no-challenge \
         "$url" -O "$out" --continue
}

# skip_note <what> <how-to-point-at-it> <expected-path>
skip_note() {
    cat <<EOF

Skipping $1.
If you already have it, place it at $3
or $2 (see README.md).
EOF
}

# md5_check <file> <expected-md5> -> warn (do not fail) if the file differs from the one used for the paper.
md5_check() {
    local got; got="$(md5sum "$1" | cut -d' ' -f1)"
    if [ "$got" = "$2" ]; then
        echo "md5 OK: $1"
    else
        echo "WARNING: $1 has md5 $got; the paper results were computed with md5 $2." >&2
    fi
}
