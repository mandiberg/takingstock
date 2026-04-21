"""
Audio batch collector — run in a separate tmux pane alongside install_make_tts.py.

On launch you are prompted to choose a mode:

  Continuous — continuous loop; waits until BATCH_SIZE .wav files accumulate,
               then zips and removes them, then waits again.
               CLI flags: --batch-size, --poll-interval

  Single     — one-shot; zips every .wav currently in tts_bark_out/ into one
               archive and exits.  Useful for a final sweep at the end of a run.

Requires the `pick` package (pip install pick).
"""

from __future__ import annotations

import argparse
import os
import shutil
import time
import zipfile

from pick import pick

_HERE = os.path.dirname(os.path.abspath(__file__))

OUT_DIR      = os.path.join(_HERE, "tts_bark_out")
DOWNLOAD_DIR = os.path.join(_HERE, "downloads")
COUNTER_FILE = os.path.join(DOWNLOAD_DIR, ".batch_counter")

BATCH_SIZE    = 200
POLL_INTERVAL = 30  # seconds


# ---------------------------------------------------------------------------
# Counter helpers
# ---------------------------------------------------------------------------

def _read_counter() -> int:
    """Return the next batch number, persisted across restarts."""
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
# Core logic
# ---------------------------------------------------------------------------

def _all_wavs(out_dir: str) -> list[str]:
    """Return all .wav files in out_dir, sorted oldest-first."""
    entries = [
        os.path.join(out_dir, fname)
        for fname in os.listdir(out_dir)
        if fname.lower().endswith(".wav")
    ]
    entries.sort(key=lambda p: os.path.getmtime(p))
    return entries


def _oldest_wavs(out_dir: str, n: int) -> list[str]:
    """Return the n oldest .wav files in out_dir, sorted by modification time."""
    return _all_wavs(out_dir)[:n]


def _make_batch_zip(files: list[str], batch_num: int, download_dir: str) -> str:
    """
    Move files into a staging dir, zip them, remove staging dir.
    Returns the path to the created zip.
    """
    label = f"batch_{batch_num:03d}"
    staging = os.path.join(download_dir, f"staging_{label}")
    zip_path = os.path.join(download_dir, f"{label}.zip")

    os.makedirs(staging, exist_ok=True)
    try:
        for src in files:
            shutil.move(src, os.path.join(staging, os.path.basename(src)))

        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for fname in os.listdir(staging):
                zf.write(os.path.join(staging, fname), arcname=fname)
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    return zip_path


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def run_continuous(batch_size: int, poll_interval: int) -> None:
    """Continuous loop: zip every batch_size files as they accumulate."""
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    print(f"Watching {OUT_DIR}")
    print(f"Batch size: {batch_size} files  |  Poll interval: {poll_interval}s")
    print(f"Zips will be written to {DOWNLOAD_DIR}")
    print("Press Ctrl-C to stop.\n")

    while True:
        wavs = _oldest_wavs(OUT_DIR, batch_size)
        if len(wavs) < batch_size:
            remaining = batch_size - len(wavs)
            print(f"  {len(wavs)} file(s) ready — waiting for {remaining} more …", flush=True)
            time.sleep(poll_interval)
            continue

        batch_num = _read_counter()
        zip_path = _make_batch_zip(wavs, batch_num, DOWNLOAD_DIR)
        _write_counter(batch_num + 1)

        label = f"batch_{batch_num:03d}"
        print(f"[{label}] Zipped {len(wavs)} files → {zip_path}", flush=True)


def run_single() -> None:
    """One-shot: zip every .wav currently in the folder and exit."""
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    wavs = _all_wavs(OUT_DIR)
    if not wavs:
        print(f"No .wav files found in {OUT_DIR}. Nothing to do.")
        return

    print(f"Active mode: found {len(wavs)} file(s) in {OUT_DIR}")
    batch_num = _read_counter()
    zip_path = _make_batch_zip(wavs, batch_num, DOWNLOAD_DIR)
    _write_counter(batch_num + 1)

    label = f"batch_{batch_num:03d}"
    print(f"[{label}] Zipped {len(wavs)} files → {zip_path}")
    print("Done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Move processed WAV files from tts_bark_out/ into zip archives in downloads/."
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Number of WAV files per zip archive in passive mode (default: {BATCH_SIZE}).",
    )
    p.add_argument(
        "--poll-interval",
        type=int,
        default=POLL_INTERVAL,
        help=f"Seconds between scans in passive mode (default: {POLL_INTERVAL}).",
    )
    return p


if __name__ == "__main__":
    args = _build_argparser().parse_args()

    mode, _ = pick(
        ["Continuous — loop, zip every N files as they arrive",
         "Single     — zip everything in the folder right now and exit"],
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
