"""Generate one SQL file from reviewed dupe folders.

FOLDER contains subfolders starting with "cluster" or "solo".
 - cluster*: files in sequential order; each consecutive pair (0,1), (2,3)... are dupes,
   the first of the pair is kept.
 - solo*: all files (3+) are dupes of the first file.
Filenames end in <sortindex>_<image_id>.<ext>
"""
import os
import re

FOLDER = "/Users/michaelmandiberg/Downloads/dupe_folders_nov1"
OUTPUT_SQL = os.path.join(FOLDER, "dupes.sql")

FILE_RE = re.compile(r"^(?P<prefix>.*?)_?(?P<idx>\d+)(?P<sfx>[A-Za-z]?)_(?P<image_id>\d+)\.[A-Za-z0-9]+$")


def parse(filename):
    m = FILE_RE.match(filename)
    if not m:
        return None
    return (int(m.group("idx")), m.group("sfx")), m.group("image_id")


def list_files(path):
    files = []
    for name in os.listdir(path):
        if name.startswith(".") or not os.path.isfile(os.path.join(path, name)):
            continue
        parsed = parse(name)
        if parsed is None:
            print(f"WARNING: cannot parse image_id from {path}/{name}, skipping")
            continue
        files.append((parsed[0], name, parsed[1]))
    files.sort()
    return [(name, image_id) for _, name, image_id in files]


def make_sql(keep_id, keep_name, dupe_id):
    return (f"UPDATE Encodings SET is_dupe_of = {keep_id} WHERE image_id = {dupe_id}; \n"
            f" -- image_id_i is {keep_id} with filename {keep_name}")


def main():
    statements = []
    for folder in sorted(os.listdir(FOLDER)):
        path = os.path.join(FOLDER, folder)
        if not os.path.isdir(path):
            continue
        files = list_files(path)
        if folder.startswith("cluster"):
            if len(files) % 2:
                print(f"WARNING: {folder} has odd number of files ({len(files)}); last one ignored")
            pairs = [(files[k], files[k + 1]) for k in range(0, len(files) - 1, 2)]
            for (keep_name, keep_id), (_, dupe_id) in pairs:
                statements.append(make_sql(keep_id, keep_name, dupe_id))
        elif folder.startswith("solo"):
            if len(files) < 2:
                continue
            keep_name, keep_id = files[0]
            for _, dupe_id in files[1:]:
                statements.append(make_sql(keep_id, keep_name, dupe_id))

    with open(OUTPUT_SQL, "w") as f:
        f.write("\n".join(statements) + "\n")
    print(f"Wrote {len(statements)} statements to {OUTPUT_SQL}")


if __name__ == "__main__":
    main()
