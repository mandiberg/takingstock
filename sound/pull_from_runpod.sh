#!/usr/bin/env bash
# pull_from_runpod.sh
#
# Downloads all batch_*.zip files from the RunPod downloads/ folder,
# then deletes each one from RunPod after a successful transfer.
# Also downloads have_barked.csv from the parent directory (always, even
# if there are no zip files ready yet).
# Runs continuously, polling RunPod once per hour.
# Each cycle:
#   - Downloads all batch_*.zip files from the RunPod downloads/ folder,
#     then deletes each one from RunPod after a successful transfer.
#   - Downloads have_barked.csv from the parent directory (always, even
#     if there are no zip files ready yet).
#
# Usage:
#   bash pull_from_runpod.sh
#
# Update the connection variables below each RunPod session (IP and port change).
# Press Ctrl-C to stop.

set -euo pipefail
set -uo pipefail

# -----------------------------------------------------------------------
# Connection — update these each RunPod session
# -----------------------------------------------------------------------
RUNPOD_KEY="$HOME/.ssh/id_ed25519"
RUNPOD_USER="root"
RUNPOD_HOST="213.173.98.69"
RUNPOD_PORT="48304"
RUNPOD_HOST="203.57.40.109"
RUNPOD_PORT="10068"
REMOTE_DIR="/root/install_make_tts/downloads"
LOCAL_DIR="/Users/tenchc/Documents/GitHub/taking_stock_production/tts_downloads"
POLL_INTERVAL=3600   # seconds between polls (1 hour)
# -----------------------------------------------------------------------

REMOTE_PARENT="$(dirname "$REMOTE_DIR")"
@ -31,67 +35,80 @@ SCP_OPTS=(-i "$RUNPOD_KEY" -P "$RUNPOD_PORT" -o StrictHostKeyChecking=no -o Batc

mkdir -p "$LOCAL_DIR"

echo "Connecting to ${RUNPOD_USER}@${RUNPOD_HOST}:${RUNPOD_PORT}"
echo "Remote zips: ${REMOTE_DIR}"
echo "Remote CSV:  ${REMOTE_PARENT}/have_barked.csv"
echo "Local:       ${LOCAL_DIR}"
echo "Polling ${RUNPOD_USER}@${RUNPOD_HOST}:${RUNPOD_PORT} every $((POLL_INTERVAL / 60)) minute(s)."
echo "Local destination: ${LOCAL_DIR}"
echo "Press Ctrl-C to stop."
echo ""

# ---------------------------------------------------------------------------
# 1. Zip files
# ---------------------------------------------------------------------------
REMOTE_FILES=()
while IFS= read -r line; do
    [[ -n "$line" ]] && REMOTE_FILES+=("$line")
done < <(
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
run_once() {
    local timestamp
    timestamp="$(date '+%Y-%m-%d %H:%M:%S')"
    echo "========================================"
    echo "  Poll started: ${timestamp}"
    echo "========================================"

    # -----------------------------------------------------------------------
    # 1. Zip files
    # -----------------------------------------------------------------------
    REMOTE_FILES=()
    while IFS= read -r line; do
        [[ -n "$line" ]] && REMOTE_FILES+=("$line")
    done < <(
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
            echo -n "Pulling ${filename} ... "
            if scp "${SCP_OPTS[@]}" \
                    "${RUNPOD_USER}@${RUNPOD_HOST}:${remote_path}" \
                    "${LOCAL_DIR}/${filename}"; then
                echo "OK"
                echo -n "  Removing from RunPod ... "
                ssh "${SSH_OPTS[@]}" "${RUNPOD_USER}@${RUNPOD_HOST}" "rm -f '${remote_path}'"
                echo "done"
                (( pulled++ )) || true
            else
                echo "FAILED (remote file left untouched)"
                (( skipped++ )) || true
            fi
        done

        echo ""
        echo "Zips: ${pulled} pulled, ${skipped} skipped."
    fi

    # -----------------------------------------------------------------------
    # 2. have_barked.csv — always pulled, never deleted (it's a running log)
    # -----------------------------------------------------------------------
    echo ""
    echo "Zips: ${pulled} pulled, ${skipped} skipped."
fi
    echo -n "Pulling have_barked.csv ... "
    if scp "${SCP_OPTS[@]}" \
            "${RUNPOD_USER}@${RUNPOD_HOST}:${REMOTE_PARENT}/have_barked.csv" \
            "${LOCAL_DIR}/have_barked.csv"; then
        echo "OK"
    else
        echo "FAILED (file may not exist yet)"
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
    echo "Done. Next poll in $((POLL_INTERVAL / 60)) minute(s) (at $(date -v +${POLL_INTERVAL}S '+%H:%M:%S'))."
    echo ""
}

echo ""
echo "Done. Files saved to: ${LOCAL_DIR}"
while true; do
    run_once
    sleep "$POLL_INTERVAL"
done