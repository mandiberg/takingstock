"""
Audio batch collector — run in a separate tmux pane alongside install_make_tts.py.

Polls tts_bark_out/ and, every BATCH_SIZE .wav files, moves them into a
sequentially-numbered zip archive inside downloads/ for easy transfer back
to your local machine.

Usage:
    python batch_collector.py                        # defaults
    python batch_collector.py --batch-size 100 --poll-interval 60
"""

from __future__ import annotations

import argparse
import os
import shutil
import time
import zipfile

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

def _oldest_wavs(out_dir: str, n: int) -> list[str]:
    """Return the n oldest .wav files in out_dir, sorted by modification time."""
    entries = [
        os.path.join(out_dir, fname)
        for fname in os.listdir(out_dir)
        if fname.lower().endswith(".wav")
    ]
    entries.sort(key=lambda p: os.path.getmtime(p))
    return entries[:n]


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


def run(batch_size: int, poll_interval: int) -> None:
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
        help=f"Number of WAV files per zip archive (default: {BATCH_SIZE}).",
    )
    p.add_argument(
        "--poll-interval",
        type=int,
        default=POLL_INTERVAL,
        help=f"Seconds between scans of tts_bark_out/ (default: {POLL_INTERVAL}).",
    )
    return p


if __name__ == "__main__":
    args = _build_argparser().parse_args()
    try:
        run(batch_size=args.batch_size, poll_interval=args.poll_interval)
    except KeyboardInterrupt:
        print("\nStopped.")
