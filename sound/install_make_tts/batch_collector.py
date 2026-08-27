"""
Audio batch collector — run in a separate tmux pane alongside install_make_coqui.py.

On launch you are prompted to choose a mode:

  Continuous — continuous loop; waits until BATCH_SIZE .wav files accumulate,
               then sorts them into a two-level MD5 hash folder tree, zips the
               tree, and removes the originals, then waits again.
               CLI flags: --batch-size, --poll-interval

  Single     — one-shot; hash-sorts and zips every .wav currently in
               tts_bark_out/ into one archive and exits.  Useful for a final
               sweep at the end of a run.

Hash key: full filename (including extension), e.g. "84231_coqui_p263.wav".
Zip layout matches DataIO.get_hash_folders() / make_tts.py:
  <L1>/<L1L2>/filename.wav   (256 leaf folders)

Requires the `pick` package (pip install pick).
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import time
import zipfile

from pick import pick

_HERE = os.path.dirname(os.path.abspath(__file__))

OUT_DIR      = os.path.join(_HERE, "tts_bark_out")
DOWNLOAD_DIR = os.path.join(_HERE, "downloads")
COUNTER_FILE = os.path.join(DOWNLOAD_DIR, ".batch_counter")

BATCH_SIZE    = 1000
POLL_INTERVAL = 600  # seconds


# ---------------------------------------------------------------------------
# Counter helpers
# ---------------------------------------------------------------------------

def _read_counter() -> int:
    if os.path.exists(COUNTER_FILE):
        try:
            return int(open(COUNTER_FILE).read().strip())
        except (ValueError, OSError):
            pass
    return 1


def _write_counter(n: int) -> None:
    with open(COUNTER_FILE, "w") as f:
        f.write(str(n))


# ---------------------------------------------------------------------------
# Hash helpers
# ---------------------------------------------------------------------------

def _get_hash_folders(filename: str) -> tuple[str, str]:
    """Derive (level1, level2) folder names from MD5 of the full filename.

    level1  → first hex char uppercased        e.g. 'A'
    level2  → first two hex chars uppercased   e.g. 'AB'

    Hash key is the complete filename including extension so that the layout
    matches make_tts.py and DataIO.get_hash_folders() on the full filename.
    """
    d = hashlib.md5(filename.encode("utf-8")).hexdigest()
    return d[0].upper(), d[0:2].upper()


# ---------------------------------------------------------------------------
# WAV discovery
# ---------------------------------------------------------------------------

def _all_wavs(out_dir: str) -> list[str]:
    entries = [
        os.path.join(out_dir, fname)
        for fname in os.listdir(out_dir)
        if fname.lower().endswith(".wav")
    ]
    entries.sort(key=lambda p: os.path.getmtime(p))
    return entries


def _oldest_wavs(out_dir: str, n: int) -> list[str]:
    return _all_wavs(out_dir)[:n]


# ---------------------------------------------------------------------------
# Core zip builder (hash-sorted)
# ---------------------------------------------------------------------------

def _make_batch_zip(files: list[str], batch_num: int, download_dir: str) -> str:
    """
    Move files into a two-level MD5 hash folder tree inside a staging dir,
    zip the entire tree (preserving folder structure), then remove staging.

    Zip internal layout:
        <L1>/<L1L2>/filename.wav
    """
    label   = f"batch_{batch_num:03d}"
    staging = os.path.join(download_dir, f"staging_{label}")
    zip_path = os.path.join(download_dir, f"{label}.zip")

    try:
        for src in files:
            fname = os.path.basename(src)
            l1, l2 = _get_hash_folders(fname)
            dest_folder = os.path.join(staging, l1, l2)
            os.makedirs(dest_folder, exist_ok=True)
            shutil.move(src, os.path.join(dest_folder, fname))

        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for root, _dirs, fnames in os.walk(staging):
                for fn in sorted(fnames):
                    fp = os.path.join(root, fn)
                    arcname = os.path.relpath(fp, staging)
                    zf.write(fp, arcname=arcname)
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    return zip_path


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def run_continuous(batch_size: int, poll_interval: int) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    print(f"Watching {OUT_DIR}")
    print(f"Batch size: {batch_size} files  |  Poll interval: {poll_interval}s")
    print(f"Zips will be written to {DOWNLOAD_DIR}")
    print("Hash key: full filename (MD5, two-level folder tree preserved in zip)")
    print("Press Ctrl-C to stop.\n")

    while True:
        wavs = _oldest_wavs(OUT_DIR, batch_size)
        if len(wavs) < batch_size:
            remaining = batch_size - len(wavs)
            print(f"  {len(wavs)} file(s) ready — waiting for {remaining} more …", flush=True)
            time.sleep(poll_interval)
            continue

        batch_num = _read_counter()
        zip_path  = _make_batch_zip(wavs, batch_num, DOWNLOAD_DIR)
        _write_counter(batch_num + 1)

        label = f"batch_{batch_num:03d}"
        print(f"[{label}] Hash-sorted and zipped {len(wavs)} files → {zip_path}", flush=True)


def run_single() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    wavs = _all_wavs(OUT_DIR)
    if not wavs:
        print(f"No .wav files found in {OUT_DIR}. Nothing to do.")
        return

    print(f"Single mode: found {len(wavs)} file(s) in {OUT_DIR}")
    batch_num = _read_counter()
    zip_path  = _make_batch_zip(wavs, batch_num, DOWNLOAD_DIR)
    _write_counter(batch_num + 1)

    label = f"batch_{batch_num:03d}"
    print(f"[{label}] Hash-sorted and zipped {len(wavs)} files → {zip_path}")
    print("Done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Hash-sort and zip processed WAV files from tts_bark_out/ "
            "into archives in downloads/. Files are organised into a two-level "
            "MD5 hash folder tree (keyed on the full filename) inside each zip."
        )
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Number of WAV files per zip archive in continuous mode (default: {BATCH_SIZE}).",
    )
    p.add_argument(
        "--poll-interval",
        type=int,
        default=POLL_INTERVAL,
        help=f"Seconds between scans in continuous mode (default: {POLL_INTERVAL}).",
    )
    return p


if __name__ == "__main__":
    args = _build_argparser().parse_args()

    mode, _ = pick(
        ["Continuous — loop, hash-sort and zip every N files as they arrive",
         "Single     — hash-sort and zip everything in the folder right now and exit"],
        "Select batch_collector mode:",
        indicator="→",
    )

    try:
        if mode.startswith("Single"):
            run_single()
        else:
            run_continuous(batch_size=args.batch_size, poll_interval=args.poll_interval)
    except KeyboardInterrupt:
        print("\nStopped.")
