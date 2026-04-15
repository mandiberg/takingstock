#!/usr/bin/env bash
# pull_from_runpod.sh
#
# Downloads all batch_*.zip files from the RunPod downloads/ folder,
# then deletes each one from RunPod after a successful transfer.
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

SSH_OPTS=(-i "$RUNPOD_KEY" -p "$RUNPOD_PORT" -o StrictHostKeyChecking=no -o BatchMode=yes)
SCP_OPTS=(-i "$RUNPOD_KEY" -P "$RUNPOD_PORT" -o StrictHostKeyChecking=no -o BatchMode=yes)

mkdir -p "$LOCAL_DIR"

echo "Connecting to ${RUNPOD_USER}@${RUNPOD_HOST}:${RUNPOD_PORT}"
echo "Remote: ${REMOTE_DIR}"
echo "Local:  ${LOCAL_DIR}"
echo ""

# Collect the list of batch zips from the remote
mapfile -t REMOTE_FILES < <(
    ssh "${SSH_OPTS[@]}" "${RUNPOD_USER}@${RUNPOD_HOST}" \
        "ls ${REMOTE_DIR}/batch_*.zip 2>/dev/null || true"
)

if [[ ${#REMOTE_FILES[@]} -eq 0 ]] || [[ -z "${REMOTE_FILES[0]}" ]]; then
    echo "No batch_*.zip files found in ${REMOTE_DIR}. Nothing to do."
    exit 0
fi

echo "Found ${#REMOTE_FILES[@]} file(s) to pull:"
for f in "${REMOTE_FILES[@]}"; do echo "  $f"; done
echo ""

pulled=0
skipped=0

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
echo "Summary: ${pulled} pulled, ${skipped} skipped."
echo "Zips saved to: ${LOCAL_DIR}"

# Download have_barked.csv from one directory up (parent of downloads/)
REMOTE_PARENT="$(dirname "$REMOTE_DIR")"
CSV_FILE="have_barked.csv"
echo ""
echo -n "Pulling ${CSV_FILE} from ${REMOTE_PARENT} ... "
if scp "${SCP_OPTS[@]}" \
        "${RUNPOD_USER}@${RUNPOD_HOST}:${REMOTE_PARENT}/${CSV_FILE}" \
        "${LOCAL_DIR}/${CSV_FILE}"; then
    echo "OK"
else
    echo "FAILED (file may not exist yet)"
fi
