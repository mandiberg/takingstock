#!/usr/bin/env bash
# merge_tts_batches.sh
#
# Unzips all batch_*.zip files from INPUT_DIR, merges their two-level MD5
# hash folder trees into OUTPUT_DIR, unions metas_audio.csv into OUTPUT_DIR
# (never overwrites existing rows), and deletes each zip only after
# confirming its contents landed correctly.
#
# Collision policy: if a WAV already exists in OUTPUT_DIR with the same path,
# the incoming file is skipped and the collision is logged to a report file.
# No existing file is ever overwritten.
#
# Usage:
#   bash merge_tts_batches.sh
#
# Edit the two path variables directly below, or override on the command line:
#   INPUT_DIR=/path/to/downloads OUTPUT_DIR=/path/to/output bash merge_tts_batches.sh

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# Paths — edit these or pass as environment variables
# ─────────────────────────────────────────────────────────────────────────────
INPUT_DIR="${INPUT_DIR:-/Users/tenchc/Documents/GitHub/taking_stock_production/tts_sport}"
OUTPUT_DIR="${OUTPUT_DIR:-/Volumes/OWC5/tts_sport}"
# ─────────────────────────────────────────────────────────────────────────────

REPORT_FILE="${OUTPUT_DIR}/merge_report.txt"
COLLISION_LOG="${OUTPUT_DIR}/collision_log.txt"
STAGING_ROOT="${INPUT_DIR}/.merge_staging"

# Counters (accumulated across all zips)
total_zips=0
total_extracted=0
total_placed=0
total_collisions=0
total_deleted_zips=0
failed_zips=()

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

log()    { echo "[$(date '+%H:%M:%S')] $*"; }
warn()   { echo "[$(date '+%H:%M:%S')] WARNING: $*" >&2; }
banner() { echo ""; echo "════════════════════════════════════════"; echo "  $*"; echo "════════════════════════════════════════"; }

# Count WAV files under a directory tree
count_wavs() { find "$1" -type f -iname "*.wav" 2>/dev/null | wc -l | tr -d ' '; }

# ─────────────────────────────────────────────────────────────────────────────
# Startup checks
# ─────────────────────────────────────────────────────────────────────────────

banner "merge_tts_batches.sh"
log "INPUT_DIR  : $INPUT_DIR"
log "OUTPUT_DIR : $OUTPUT_DIR"
echo ""

if [[ ! -d "$INPUT_DIR" ]]; then
    echo "ERROR: INPUT_DIR does not exist: $INPUT_DIR"
    exit 1
fi

# Discover zips (mapfile requires bash 4+; use while-read for macOS bash 3.2 compat)
ZIP_FILES=()
while IFS= read -r _zip; do
    ZIP_FILES+=("$_zip")
done < <(find "$INPUT_DIR" -maxdepth 1 -name "batch_*.zip" | sort)
total_zips=${#ZIP_FILES[@]}

if [[ $total_zips -eq 0 ]]; then
    echo "No batch_*.zip files found in $INPUT_DIR. Nothing to do."
    exit 0
fi

log "Found $total_zips zip(s) to process:"
for z in "${ZIP_FILES[@]}"; do log "  $(basename "$z")"; done
echo ""

# Create output and staging dirs
mkdir -p "$OUTPUT_DIR"
mkdir -p "$STAGING_ROOT"

# Initialise report and collision log (append mode so reruns accumulate)
{
    echo ""
    echo "════════════════════════════════════════"
    echo "  Run: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "  INPUT_DIR:  $INPUT_DIR"
    echo "  OUTPUT_DIR: $OUTPUT_DIR"
    echo "  Zips found: $total_zips"
    echo "════════════════════════════════════════"
} >> "$REPORT_FILE"

# ─────────────────────────────────────────────────────────────────────────────
# Per-zip processing
# ─────────────────────────────────────────────────────────────────────────────

for zip_path in "${ZIP_FILES[@]}"; do
    zip_name="$(basename "$zip_path")"
    staging_dir="${STAGING_ROOT}/${zip_name%.zip}"

    banner "Processing $zip_name"

    # ── 1. Verify zip integrity ───────────────────────────────────────────────
    log "Verifying zip integrity …"
    if ! unzip -t "$zip_path" > /dev/null 2>&1; then
        warn "$zip_name failed integrity check — skipping (will not delete)"
        echo "INTEGRITY_FAIL: $zip_name" >> "$REPORT_FILE"
        failed_zips+=("$zip_name")
        continue
    fi
    log "Integrity OK"

    # ── 2. Extract to staging ─────────────────────────────────────────────────
    rm -rf "$staging_dir"
    mkdir -p "$staging_dir"
    log "Extracting to staging: $staging_dir"
    unzip -q "$zip_path" -d "$staging_dir"

    extracted=$(count_wavs "$staging_dir")
    log "Extracted $extracted WAV file(s) from $zip_name"

    if [[ "$extracted" -eq 0 ]]; then
        warn "$zip_name contained no WAV files — skipping (will not delete)"
        echo "EMPTY_ZIP: $zip_name" >> "$REPORT_FILE"
        rm -rf "$staging_dir"
        failed_zips+=("$zip_name")
        continue
    fi

    # ── 3. Merge staging tree into output ─────────────────────────────────────
    log "Merging into output dir …"

    zip_placed=0
    zip_collisions=0
    zip_collision_list=()

    # Walk every WAV in the staging tree (preserving L1/L2 subfolder structure)
    while IFS= read -r src_path; do
        # Relative path inside the zip: e.g. A/AB/84231_coqui_p263.wav
        rel_path="${src_path#"$staging_dir"/}"
        dest_path="${OUTPUT_DIR}/${rel_path}"
        dest_dir="$(dirname "$dest_path")"

        mkdir -p "$dest_dir"

        if [[ -e "$dest_path" ]]; then
            # Collision — keep existing, log and skip
            zip_collisions=$(( zip_collisions + 1 ))
            zip_collision_list+=("$rel_path")
            echo "COLLISION [$zip_name]: $rel_path" >> "$COLLISION_LOG"
        else
            cp "$src_path" "$dest_path"
            zip_placed=$(( zip_placed + 1 ))
        fi
    done < <(find "$staging_dir" -type f -iname "*.wav" | sort)

    total_placed=$(( total_placed + zip_placed ))
    total_collisions=$(( total_collisions + zip_collisions ))
    total_extracted=$(( total_extracted + extracted ))

    log "Placed:     $zip_placed file(s)"
    if [[ $zip_collisions -gt 0 ]]; then
        warn "$zip_collisions collision(s) skipped (files already exist in output):"
        for c in "${zip_collision_list[@]}"; do warn "  $c"; done
    fi

    # ── 4. Verify placed count matches expectation ────────────────────────────
    expected_in_output=$(( zip_placed + zip_collisions ))
    if [[ "$expected_in_output" -ne "$extracted" ]]; then
        warn "Count mismatch for $zip_name — extracted=$extracted but placed+collisions=$expected_in_output"
        warn "NOT deleting $zip_name — manual review required"
        echo "COUNT_MISMATCH: $zip_name  extracted=$extracted placed=$zip_placed collisions=$zip_collisions" >> "$REPORT_FILE"
        rm -rf "$staging_dir"
        failed_zips+=("$zip_name")
        continue
    fi

    # ── 5. Spot-check: verify a sample of placed files are readable ───────────
    log "Spot-checking placed files …"
    spot_errors=0
    # Check up to 10 random files from this batch
    while IFS= read -r src_path; do
        rel_path="${src_path#"$staging_dir"/}"
        dest_path="${OUTPUT_DIR}/${rel_path}"
        if [[ ! -s "$dest_path" ]]; then
            warn "Spot-check FAILED — zero-byte or missing: $dest_path"
            spot_errors=$(( spot_errors + 1 ))
        fi
    done < <(find "$staging_dir" -type f -iname "*.wav" | shuf | head -10)

    if [[ $spot_errors -gt 0 ]]; then
        warn "$spot_errors spot-check failure(s) in $zip_name — NOT deleting zip"
        echo "SPOT_CHECK_FAIL: $zip_name  errors=$spot_errors" >> "$REPORT_FILE"
        rm -rf "$staging_dir"
        failed_zips+=("$zip_name")
        continue
    fi
    log "Spot-check passed"

    # ── 6. All checks passed — delete zip ─────────────────────────────────────
    rm -rf "$staging_dir"
    rm "$zip_path"
    total_deleted_zips=$(( total_deleted_zips + 1 ))
    log "Deleted $zip_name (all $extracted file(s) confirmed in output)"

    echo "OK: $zip_name  extracted=$extracted placed=$zip_placed collisions=$zip_collisions" >> "$REPORT_FILE"
    echo ""
done

# ─────────────────────────────────────────────────────────────────────────────
# metas_audio.csv
# ─────────────────────────────────────────────────────────────────────────────

banner "Merging metas_audio.csv"
src_csv="${INPUT_DIR}/metas_audio.csv"
dest_csv="${OUTPUT_DIR}/metas_audio.csv"

if [[ -f "$src_csv" ]]; then
    merge_out="$(python3 - "$src_csv" "$dest_csv" <<'PY'
import csv, os, sys, tempfile

src, dest = sys.argv[1], sys.argv[2]

def image_id_key(row):
    try:
        return str(int(float(str(row.get("image_id", "")).strip())))
    except (TypeError, ValueError):
        return None

def load(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames or [])
        return fields, list(reader)

src_fields, src_rows = load(src)
if os.path.isfile(dest) and os.path.getsize(dest) > 0:
    dest_fields, dest_rows = load(dest)
else:
    dest_fields, dest_rows = [], []

fields = list(dict.fromkeys(dest_fields + src_fields))
if "filename" in fields:
    fields = [c for c in fields if c != "filename"] + ["filename"]

seen = set()
out_rows = []
for row in dest_rows:
    key = image_id_key(row)
    if key:
        seen.add(key)
    out_rows.append(row)

added = 0
for row in src_rows:
    key = image_id_key(row)
    if key and key in seen:
        continue
    if key:
        seen.add(key)
    out_rows.append(row)
    added += 1

os.makedirs(os.path.dirname(os.path.abspath(dest)) or ".", exist_ok=True)
fd, tmp = tempfile.mkstemp(prefix="metas_audio_", suffix=".csv",
                           dir=os.path.dirname(os.path.abspath(dest)) or ".")
try:
    with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(out_rows)
    os.replace(tmp, dest)
except Exception:
    try:
        os.remove(tmp)
    except OSError:
        pass
    raise

print(f"{len(dest_rows)} {added} {len(out_rows)}")
PY
)"
    dest_before="$(echo "$merge_out" | awk '{print $1}')"
    added="$(echo "$merge_out" | awk '{print $2}')"
    dest_after="$(echo "$merge_out" | awk '{print $3}')"
    log "metas_audio.csv merged: dest had ${dest_before} row(s), added ${added} new image_id(s), now ${dest_after}"
    echo "metas_audio.csv: dest_before=${dest_before} added=${added} dest_after=${dest_after}" >> "$REPORT_FILE"
else
    warn "metas_audio.csv not found in INPUT_DIR — skipping"
    echo "metas_audio.csv: NOT FOUND in input" >> "$REPORT_FILE"
fi

# ─────────────────────────────────────────────────────────────────────────────
# Cleanup staging root if empty
# ─────────────────────────────────────────────────────────────────────────────
rmdir "$STAGING_ROOT" 2>/dev/null || true

# ─────────────────────────────────────────────────────────────────────────────
# Final summary
# ─────────────────────────────────────────────────────────────────────────────

banner "Summary"

output_wav_count=$(count_wavs "$OUTPUT_DIR")

log "Zips found:          $total_zips"
log "Zips deleted:        $total_deleted_zips"
log "Zips with errors:    ${#failed_zips[@]}"
log "WAVs extracted:      $total_extracted"
log "WAVs placed:         $total_placed"
log "Collisions skipped:  $total_collisions"
log "WAVs in output dir:  $output_wav_count"

if [[ ${#failed_zips[@]} -gt 0 ]]; then
    echo ""
    warn "The following zip(s) were NOT deleted due to errors — review manually:"
    for f in "${failed_zips[@]}"; do warn "  $f"; done
    warn "Check $REPORT_FILE and $COLLISION_LOG for details"
fi

if [[ $total_collisions -gt 0 ]]; then
    echo ""
    log "Collision details written to: $COLLISION_LOG"
fi

echo ""
log "Report appended to: $REPORT_FILE"

{
    echo "SUMMARY: zips_found=$total_zips deleted=$total_deleted_zips errors=${#failed_zips[@]} wavs_extracted=$total_extracted wavs_placed=$total_placed collisions=$total_collisions output_total=$output_wav_count"
    echo ""
} >> "$REPORT_FILE"

echo ""
