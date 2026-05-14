import os
import hashlib
import shutil
import argparse

# ── SET YOUR OUTPUT DIRECTORY HERE ──────────────────────────────────────────
OUTPUT_DIR = "/Users/tenchc/Desktop/Hashing_Test"
# ────────────────────────────────────────────────────────────────────────────

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".aac", ".m4a", ".aiff", ".aif"}
HASH_ALPHABET = list("ABCDEF0123456789")


def get_hash_folders(hash_key):
    """Return (level1, level2) folder names derived from MD5 of hash_key.

    Mirrors DataIO.get_hash_folders() in mp_db_io.py.
    level1  → first hex char uppercased        e.g. '3'
    level2  → first two hex chars uppercased   e.g. '3B'
    """
    m = hashlib.md5()
    m.update(hash_key.encode("utf-8"))
    d = m.hexdigest()
    return d[0].upper(), d[0:2].upper()


def make_hash_folders(path):
    """Create the full two-level (16×16 = 256 leaf) hash folder tree under path.

    Mirrors DataIO.make_hash_folders() in mp_db_io.py.
    Structure: path/<L1>/<L1L2>/
    """
    for letter in HASH_ALPHABET:
        for letter2 in HASH_ALPHABET:
            leaf = os.path.join(path, letter, letter + letter2)
            os.makedirs(leaf, exist_ok=True)


def extract_hash_key(filename):
    """Split filename at the first '_' and return the prefix as the hash key.

    Example: '14692993_coqui_p336.wav' → '14692993'
    If there is no '_', the full stem is used.
    """
    stem = os.path.splitext(filename)[0]
    return stem.split("_")[0]


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Move audio files from INPUT_DIR into a two-level MD5 hash folder "
            "structure under OUTPUT_DIR. The hash key is the portion of the "
            "filename before the first '_'."
        )
    )
    parser.add_argument(
        "input_dir",
        help="Directory containing audio files to move.",
    )
    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(OUTPUT_DIR)

    if not os.path.isdir(input_dir):
        print(f"ERROR: input_dir does not exist or is not a directory: {input_dir}")
        raise SystemExit(1)

    if output_dir == input_dir:
        print("ERROR: OUTPUT_DIR and input_dir must not be the same path.")
        raise SystemExit(1)

    print(f"Input  : {input_dir}")
    print(f"Output : {output_dir}")
    print("Building hash folder tree…")
    make_hash_folders(output_dir)
    print("Hash folder tree ready.")

    moved = 0
    skipped = 0

    for entry in sorted(os.scandir(input_dir), key=lambda e: e.name):
        if not entry.is_file():
            continue

        filename = entry.name

        if filename.startswith("."):
            continue

        ext = os.path.splitext(filename)[1].lower()
        if ext not in AUDIO_EXTENSIONS:
            print(f"  SKIP (not audio): {filename}")
            skipped += 1
            continue

        hash_key = extract_hash_key(filename)
        level1, level2 = get_hash_folders(hash_key)
        dest_folder = os.path.join(output_dir, level1, level2)
        dest_path = os.path.join(dest_folder, filename)

        if os.path.exists(dest_path):
            print(f"  SKIP (already exists): {filename}")
            skipped += 1
            continue

        shutil.move(entry.path, dest_path)
        print(f"  MOVED: {filename}  →  {level1}/{level2}/")
        moved += 1

    print(f"\nDone. Moved: {moved}  |  Skipped: {skipped}")


if __name__ == "__main__":
    main()
