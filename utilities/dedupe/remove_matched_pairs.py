#!/usr/bin/env python3
"""
Remove duplicate score-rating folders that contain the exact same pair of images.

Folder structure:
  <root>/
    <cluster>/
      high/ or medium/
        <score_rating>/   ← contains 2 jpgs + 1 sql
          imageA.jpg
          imageB.jpg
          dupe_*.sql

Usage:
  python remove_matched_pairs.py <root_dir>

Tracks every unique (frozenset of jpg filenames) seen globally across all
clusters and all high/medium tiers. If the exact same pair is encountered
again anywhere, the duplicate folder is deleted and a message is printed.
"""

import os
import sys
import shutil


def get_jpg_pair(score_dir: str) -> frozenset | None:
    """Return a frozenset of jpg basenames found in a score-rating folder."""
    try:
        names = [f for f in os.listdir(score_dir) if f.lower().endswith(".jpg")]
    except NotADirectoryError:
        return None
    if len(names) != 2:
        return None
    return frozenset(names)


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <root_dir>")
        sys.exit(1)

    root = sys.argv[1]
    if not os.path.isdir(root):
        print(f"Error: '{root}' is not a directory.")
        sys.exit(1)

    seen: dict[frozenset, str] = {}  # pair → first folder path that had it
    total_deleted = 0

    for cluster_name in sorted(os.listdir(root)):
        cluster_path = os.path.join(root, cluster_name)
        if not os.path.isdir(cluster_path) or cluster_name.startswith("."):
            continue

        for tier in sorted(os.listdir(cluster_path)):
            tier_path = os.path.join(cluster_path, tier)
            if not os.path.isdir(tier_path) or tier.startswith("."):
                continue

            for score_dir_name in sorted(os.listdir(tier_path)):
                score_path = os.path.join(tier_path, score_dir_name)
                if not os.path.isdir(score_path) or score_dir_name.startswith("."):
                    continue

                pair = get_jpg_pair(score_path)
                if pair is None:
                    continue

                if pair in seen:
                    images = sorted(pair)
                    print(
                        f"PERFECT MATCH — deleting duplicate:\n"
                        f"  kept:    {seen[pair]}\n"
                        f"  deleted: {score_path}\n"
                        f"  images:  {images[0]}  &  {images[1]}\n"
                    )
                    shutil.rmtree(score_path)
                    total_deleted += 1
                else:
                    seen[pair] = score_path

    print(f"Done. {total_deleted} duplicate folder(s) removed.")


if __name__ == "__main__":
    main()
 