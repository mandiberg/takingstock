'''
This script lists the contents of a FOLDER
compares the names and removes any duplicates
it should differentiate between files and folders, and only look for duplicates that have the same type (file or folder)

'''

import os
import shutil

FOLDER = "/Users/michaelmandiberg/Documents/projects-active/facemap_production/dupestuff/dedupe_this/dupe_sorting_done"
NEW_FOLDER = os.path.join(FOLDER, "deduped")
if not os.path.exists(NEW_FOLDER):
    os.makedirs(NEW_FOLDER)

# Safety defaults:
# - DRY_RUN=True means nothing is deleted/moved.
# - ACTION="report" only prints duplicates.
# - KEY_STRATEGY controls how duplicates are detected.
DRY_RUN = False
ACTION = "delete"  # "report" | "move" | "delete"
KEY_STRATEGY = "drop_last_underscore"  # "exact" | "drop_last_underscore"


def build_dedupe_key(name):
    if KEY_STRATEGY == "exact":
        return name
    if KEY_STRATEGY == "drop_last_underscore":
        parts = name.split("_")
        return "_".join(parts[:-1]) if len(parts) > 1 else name
    raise ValueError(f"Unknown KEY_STRATEGY: {KEY_STRATEGY}")


def remove_entry(path, entry_type):
    if entry_type == "file":
        os.remove(path)
    else:
        shutil.rmtree(path)

def dedupe_folder():
    # Use (entry_type, dedupe_name) so files and folders are deduped independently.
    seen = {}
    for entry in os.scandir(FOLDER):
        if entry.path == NEW_FOLDER:
            continue

        if entry.is_file(follow_symlinks=False):
            entry_type = "file"
        elif entry.is_dir(follow_symlinks=False):
            entry_type = "folder"
        else:
            print(f"Skipping unsupported entry type: {entry.name}")
            continue

        dedupe_name = build_dedupe_key(entry.name)
        dedupe_key = (entry_type, dedupe_name)

        print(f"Checking {entry_type} key '{dedupe_name}' for duplicates")
        if dedupe_key in seen:
            original_name = seen[dedupe_key]
            print(f"Duplicate found: {entry.name} (first seen: {original_name})")

            if ACTION == "report":
                continue

            target_path = os.path.join(NEW_FOLDER, entry.name)
            if ACTION == "move":
                if DRY_RUN:
                    print(f"[DRY RUN] Would move duplicate to: {target_path}")
                else:
                    shutil.move(entry.path, target_path)
                    print(f"Moved duplicate {entry_type}: {entry.path} -> {target_path}")
                continue

            if ACTION != "delete":
                raise ValueError(f"Unknown ACTION: {ACTION}")

            if DRY_RUN:
                print(f"[DRY RUN] Would delete duplicate {entry_type}: {entry.path}")
                continue

            try:
                remove_entry(entry.path, entry_type)
                print(f"Deleted duplicate {entry_type}: {entry.path}")
            except PermissionError as e:
                print(f"Permission denied, could not delete {entry.path}: {e}")
        else:
            seen[dedupe_key] = entry.name

if __name__ == "__main__":
    dedupe_folder()