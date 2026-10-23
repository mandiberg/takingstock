#!/usr/bin/env python3

import os
import csv
import hashlib
from pathlib import Path

from PIL import Image, ImageOps
import numpy as np


ORIGINALS_DIR = Path("/Volumes/LaCie/stills_to_review")
RECOVERED_DIR = Path("/Volumes/LaCie/wrong_names")
OUTPUT_CSV = Path("/Volumes/LaCie/photo_matches.csv")


IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".JPG", ".JPEG",
    ".png", ".PNG",
    ".tif", ".tiff", ".TIF", ".TIFF"
}


# ------------------------------------------------------------
# Image hashing
# ------------------------------------------------------------

def dhash(image, hash_size=16):
    """
    Difference hash.
    More tolerant than an exact file hash of JPEG
    compression/resizing differences.
    """
    image = ImageOps.exif_transpose(image)
    image = image.convert("L")
    image = image.resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)

    pixels = np.asarray(image, dtype=np.int16)

    diff = pixels[:, 1:] > pixels[:, :-1]

    # Convert boolean array to a string of bits
    return diff.flatten()


def phash(image, hash_size=16, highfreq_factor=4):
    """
    Perceptual hash using a DCT.
    """
    image = ImageOps.exif_transpose(image)
    image = image.convert("L")

    size = hash_size * highfreq_factor
    image = image.resize((size, size), Image.Resampling.LANCZOS)

    pixels = np.asarray(image, dtype=np.float32)

    # 2-D DCT, implemented with scipy if available
    try:
        from scipy.fftpack import dct
    except ImportError:
        raise RuntimeError(
            "scipy is required. Install it with:\n"
            "    pip install scipy"
        )

    dct_result = dct(dct(pixels, axis=0, norm="ortho"),
                     axis=1, norm="ortho")

    low_freq = dct_result[:hash_size, :hash_size]

    # Exclude DC coefficient
    median = np.median(low_freq[1:, :])

    return low_freq > median


def hamming_distance(a, b):
    return np.count_nonzero(a != b)


def image_signature(path):
    """
    Calculate both pHash and dHash.
    """
    with Image.open(path) as im:
        return (
            phash(im),
            dhash(im)
        )


# ------------------------------------------------------------
# File discovery
# ------------------------------------------------------------

def find_images(directory):
    files = []

    for path in directory.rglob("*"):
        if path.is_file() and path.suffix in IMAGE_EXTENSIONS:
            files.append(path)

    return sorted(files)


# ------------------------------------------------------------
# Exact file hash (useful diagnostic)
# ------------------------------------------------------------

def sha256(path, chunk_size=1024 * 1024):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)

    return h.hexdigest()


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    print()
    print("Scanning originals...")
    originals = find_images(ORIGINALS_DIR)

    print(f"Found {len(originals)} original images.")

    print()
    print("Scanning recovered files...")
    recovered = find_images(RECOVERED_DIR)

    print(f"Found {len(recovered)} recovered images.")

    if not originals:
        raise RuntimeError("No original images found.")

    if not recovered:
        raise RuntimeError("No recovered images found.")

    print()
    print("Calculating image signatures for originals...")

    original_signatures = {}

    for i, path in enumerate(originals, 1):
        try:
            original_signatures[path] = image_signature(path)
        except Exception as e:
            print(f"ERROR reading {path}: {e}")

        print(f"\r  {i}/{len(originals)}", end="", flush=True)

    print()
    print()
    print("Matching recovered images...")
    print()

    rows = []

    for r_index, recovered_path in enumerate(recovered, 1):

        try:
            recovered_phash, recovered_dhash = image_signature(recovered_path)
        except Exception as e:
            print(f"ERROR reading {recovered_path}: {e}")
            continue

        candidates = []

        for original_path, (original_phash, original_dhash) in original_signatures.items():

            p_dist = hamming_distance(
                recovered_phash,
                original_phash
            )

            d_dist = hamming_distance(
                recovered_dhash,
                original_dhash
            )

            # pHash is more important; dHash is a useful secondary check.
            combined = p_dist + (d_dist * 0.5)

            candidates.append(
                (
                    combined,
                    p_dist,
                    d_dist,
                    original_path
                )
            )

        candidates.sort(key=lambda x: x[0])

        best = candidates[0]
        second = candidates[1] if len(candidates) > 1 else None

        combined, p_dist, d_dist, original_path = best

        if second:
            margin = second[0] - best[0]
        else:
            margin = None

        rows.append({
            "recovered_file": str(recovered_path),
            "original_file": str(original_path),
            "original_filename": original_path.name,
            "original_folder": original_path.parent.name,
            "combined_distance": round(combined, 2),
            "phash_distance": p_dist,
            "dhash_distance": d_dist,
            "next_best_distance": round(second[0], 2) if second else "",
            "margin_to_next_best": round(margin, 2) if margin is not None else "",
        })

        print(
            f"[{r_index:2}/{len(recovered)}] "
            f"{recovered_path.name}"
        )
        print(
            f"       -> {original_path.name}"
            f"  pHash={p_dist}"
            f"  dHash={d_dist}"
            f"  score={combined:.1f}"
            f"  margin={margin:.1f}" if margin is not None else ""
        )

    # --------------------------------------------------------
    # Write CSV
    # --------------------------------------------------------

    fieldnames = [
        "recovered_file",
        "original_file",
        "original_filename",
        "original_folder",
        "combined_distance",
        "phash_distance",
        "dhash_distance",
        "next_best_distance",
        "margin_to_next_best",
    ]

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print()
    print("=" * 70)
    print(f"Results written to:")
    print(OUTPUT_CSV)
    print("=" * 70)

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("MATCH SUMMARY")
    print()

    for row in sorted(rows, key=lambda r: float(r["combined_distance"])):

        score = float(row["combined_distance"])
        margin = (
            float(row["margin_to_next_best"])
            if row["margin_to_next_best"] != ""
            else 0
        )

        if score <= 10 and margin >= 3:
            confidence = "VERY STRONG"

        elif score <= 20 and margin >= 3:
            confidence = "STRONG"

        elif score <= 30 and margin >= 2:
            confidence = "PROBABLY"

        else:
            confidence = "CHECK"

        print(
            f"{confidence:12} "
            f"{row['recovered_file'].split('/')[-1]:45} "
            f"-> {row['original_filename']:40} "
            f"score={score:5.1f} "
            f"margin={margin:5.1f}"
        )


if __name__ == "__main__":
    main()