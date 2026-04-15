#!/usr/bin/env bash
# pull_from_runpod.sh
#
# Downloads all batch_*.zip files from the RunPod downloads/ folder,
# then deletes each one from RunPod after a successful transfer.
# Also downloads have_barked.csv from the parent directory (always, even
# if there are no zip files ready yet).
#
# Usage:
#   bash pull_from_runpod.sh
#
# Update the connection variables below each RunPod session (IP and port change).

set -euo pipefail

# -----------------------------------------------------------------------
# Connection — update these each RunPod session
# -----------------------------------------------------------------------
RUNPOD_KEY="$HOME/.ssh/id_ed25519"
RUNPOD_USER="root"
RUNPOD_HOST="103.196.86.83"
RUNPOD_PORT="16980"
REMOTE_DIR="/workspace/install_make_tts/downloads"
LOCAL_DIR="/Users/tenchc/Documents/GitHub/taking_stock_production/tts_downloads"
# -----------------------------------------------------------------------

REMOTE_PARENT="$(dirname "$REMOTE_DIR")"

SSH_OPTS=(-i "$RUNPOD_KEY" -p "$RUNPOD_PORT" -o StrictHostKeyChecking=no -o BatchMode=yes)
SCP_OPTS=(-i "$RUNPOD_KEY" -P "$RUNPOD_PORT" -o StrictHostKeyChecking=no -o BatchMode=yes)

mkdir -p "$LOCAL_DIR"

echo "Connecting to ${RUNPOD_USER}@${RUNPOD_HOST}:${RUNPOD_PORT}"
echo "Remote zips: ${REMOTE_DIR}"
echo "Remote CSV:  ${REMOTE_PARENT}/have_barked.csv"
echo "Local:       ${LOCAL_DIR}"
echo ""

# ---------------------------------------------------------------------------
# 1. Zip files
# ---------------------------------------------------------------------------
mapfile -t REMOTE_FILES < <(
    ssh "${SSH_OPTS[@]}" "${RUNPOD_USER}@${RUNPOD_HOST}" \
        "ls ${REMOTE_DIR}/batch_*.zip 2>/dev/null || true"
)

pulled=0
skipped=0

if [[ ${#REMOTE_FILES[@]} -eq 0 ]] || [[ -z "${REMOTE_FILES[0]}" ]]; then
    echo "No batch_*.zip files found in ${REMOTE_DIR}."
else
    echo "Found ${#REMOTE_FILES[@]} zip(s) to pull:"
    for f in "${REMOTE_FILES[@]}"; do echo "  $f"; done
    echo ""

    for remote_path in "${REMOTE_FILES[@]}"; do
        filename="$(basename "$remote_path")"

        echo -n "Pulling ${filename} ... "
        if scp "${SCP_OPTS[@]}" \
                "${RUNPOD_USER}@${RUNPOD_HOST}:${remote_path}" \
                "${LOCAL_DIR}/${filename}"; then
            echo "OK"
            echo -n "  Removing from RunPod ... "
            ssh "${SSH_OPTS[@]}" "${RUNPOD_USER}@${RUNPOD_HOST}" "rm -f '${remote_path}'"
            echo "done"
            (( pulled++ ))
        else
            echo "FAILED (remote file left untouched)"
            (( skipped++ ))
        fi
    done

    echo ""
    echo "Zips: ${pulled} pulled, ${skipped} skipped."
fi

# ---------------------------------------------------------------------------
# 2. have_barked.csv — always pulled, never deleted (it's a running log)
# ---------------------------------------------------------------------------
echo ""
echo -n "Pulling have_barked.csv ... "
if scp "${SCP_OPTS[@]}" \
        "${RUNPOD_USER}@${RUNPOD_HOST}:${REMOTE_PARENT}/have_barked.csv" \
        "${LOCAL_DIR}/have_barked.csv"; then
    echo "OK"
else
    echo "FAILED (file may not exist yet)"
fi

echo ""
echo "Done. Files saved to: ${LOCAL_DIR}"
