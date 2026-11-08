"""
Scan a folder of merged images and upsert crop-key entries into the
1x1 canonical keys CSV.

Folder naming convention:
    {horizontal|square|vertical}_{MULTIPLIER_DICT_key}
    e.g.  horizontal_9x16_regular_miami
          square_sq_default
          vertical_16x9_portrait_torso

Pass a named folder directly, or a container folder whose immediate
children are named folders — both are handled automatically.

Usage:
    python scan_bins_to_csv.py <folder>
"""
import os
import re
import sys

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _ROOT)
from constants_make_video import MULTIPLIER_DICT

FULL_BODY = True
if FULL_BODY: 
    matrix_family = "BodyPoses3D"
else:
    matrix_family = "ArmsPoses3D"

PREFIX_TO_COL = {
    "horizontal": "horizontal_key",
    "square":     "square_key",
    "vertical":   "vertical_key",
}
CSV_COLUMNS = ["arms_cluster_id", "object_signature_cluster_id", "horizontal_key", "square_key", "vertical_key"]
IMG_RE      = re.compile(r"merged_cluster(-?\d+)_p(\d+)")
CSV_PATH    = os.path.join(_HERE, f"1x1_canonical_keys_{matrix_family}.csv")


def _parse_folder(folder_name):
    """Return (prefix, mult_key), or None if no recognized prefix.
    Exits with an error if prefix matches but key is not in MULTIPLIER_DICT.
    Expects a folder name with this structure: {horizontal|square|vertical}_{MULTIPLIER_DICT_key}
    """

    for prefix in PREFIX_TO_COL:
        if folder_name.lower().startswith(prefix + "_"):
            mult_key = folder_name[len(prefix) + 1:]
            if mult_key not in MULTIPLIER_DICT:
                print(f"ERROR: '{mult_key}' not found in MULTIPLIER_DICT.")
                print("Valid keys:\n  " + "\n  ".join(sorted(MULTIPLIER_DICT)))
                sys.exit(1)
            return prefix, mult_key
    return None


def _scan_pairs(folder):
    """Walk folder recursively; return set of (cluster_id_int, pose_no_int)."""
    pairs = set()
    for dirpath, _, filenames in os.walk(folder):
        for fname in filenames:
            m = IMG_RE.search(fname)
            if m:
                pairs.add((int(m.group(1)), int(m.group(2))))
    return pairs


def _load_csv(csv_path):
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, dtype=str).fillna("")
        for col in CSV_COLUMNS:
            if col not in df.columns:
                df[col] = ""
        return df
    return pd.DataFrame(columns=CSV_COLUMNS)


def _upsert(df, folder, prefix, mult_key):
    """Scan folder, upsert into df for the given prefix/key. Returns (df, added, same, conflicts)."""
    target_col = PREFIX_TO_COL[prefix]
    pairs = _scan_pairs(folder)
    if not pairs:
        print(f"  (no images found in {os.path.basename(folder)}, skipping)")
        return df, 0, 0, 0

    print(f"  {os.path.basename(folder)}  →  column '{target_col}'  →  key '{mult_key}'  ({len(pairs)} pairs)")

    added, same, conflicts = 0, 0, 0
    new_rows = []
    for cluster_id, pose_no in sorted(pairs):
        mask = (
            (df["arms_cluster_id"] == str(cluster_id)) &
            (df["object_signature_cluster_id"] == str(pose_no))
        )
        if mask.any():
            existing = df.loc[mask, target_col].iloc[0]
            if existing == mult_key:
                same += 1
            elif existing == "":
                df.loc[mask, target_col] = mult_key
                added += 1
            else:
                print(f"    CONFLICT ({cluster_id}, {pose_no}): '{existing}' → '{mult_key}' (overwriting)")
                df.loc[mask, target_col] = mult_key
                conflicts += 1
        else:
            row = {col: "" for col in CSV_COLUMNS}
            row["arms_cluster_id"]             = str(cluster_id)
            row["object_signature_cluster_id"] = str(pose_no)
            row[target_col]                    = mult_key
            new_rows.append(row)
            added += 1

    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    return df, added, same, conflicts


def main():
    if len(sys.argv) < 2:
        print(f"Usage: python {os.path.basename(__file__)} <folder>")
        sys.exit(1)

    folder   = sys.argv[1].rstrip("/\\")
    csv_path = CSV_PATH

    parsed = _parse_folder(os.path.basename(folder))

    if parsed is not None:
        # single named folder
        targets = [(folder, *parsed)]
    else:
        # container: collect immediate children with recognized prefixes
        targets = []
        for child in sorted(os.listdir(folder)):
            child_path = os.path.join(folder, child)
            if not os.path.isdir(child_path):
                continue
            result = _parse_folder(child)
            if result is not None:
                targets.append((child_path, *result))
        if not targets:
            print(f"ERROR: '{os.path.basename(folder)}' has no recognized prefix and no "
                  f"children named horizontal_*, square_*, or vertical_*")
            sys.exit(1)
        print(f"Container: {os.path.basename(folder)}  ({len(targets)} named subfolder(s))")

    df = _load_csv(csv_path)
    total_added = total_same = total_conflicts = 0

    for target_folder, prefix, mult_key in targets:
        df, added, same, conflicts = _upsert(df, target_folder, prefix, mult_key)
        total_added += added
        total_same += same
        total_conflicts += conflicts

    # sort and save
    df["_c"] = pd.to_numeric(df["arms_cluster_id"], errors="coerce")
    df["_p"] = pd.to_numeric(df["object_signature_cluster_id"], errors="coerce")
    df = df.sort_values(["_c", "_p"]).drop(columns=["_c", "_p"]).reset_index(drop=True)

    df.to_csv(csv_path, index=False)
    print(f"Done : {total_added} added/filled, {total_same} unchanged, {total_conflicts} conflicts overwritten")
    print(f"CSV  : {csv_path}")


if __name__ == "__main__":
    main()
