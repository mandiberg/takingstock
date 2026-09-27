"""Compare two image folders by file presence and image dimensions.

Usage:
    python diff_image_folders.py /path/folder1 /path/folder2

If no arguments are supplied, it uses the default FOLDER1 and FOLDER2 paths below.
"""

import argparse
import os
from typing import Dict, Optional, Set, Tuple

from pymediainfo import MediaInfo


FOLDER1 = "/Volumes/LaCie/segment_images_theoffice/output_folder/merged_Arms_thoma_sept21_all/500s"
FOLDER2 = "/Volumes/LaCie/segment_images_theoffice/output_folder/_Arms_thoma_sept27-1/merged"


def list_files_recursive(folder: str) -> Dict[str, str]:
    """Return a relative-path -> absolute-path map for all files under a folder."""
    file_map: Dict[str, str] = {}
    for root, _, files in os.walk(folder):
        for name in files:
            full_path = os.path.join(root, name)
            rel_path = os.path.relpath(full_path, folder)
            file_map[rel_path.replace(os.sep, "/")] = full_path
    return file_map


def get_media_info_library_path() -> Optional[str]:
    """Return a likely MediaInfo library path for macOS installs when available."""
    candidate_paths = [
        "/opt/homebrew/Cellar/libmediainfo/24.06/lib/libmediainfo.dylib",
        "/opt/homebrew/Cellar/libmediainfo/26.01/lib/libmediainfo.dylib",
        "/opt/homebrew/opt/libmediainfo/lib/libmediainfo.dylib",
    ]
    for path in candidate_paths:
        if os.path.exists(path):
            return path
    return None


def get_image_dimensions(file_path: str) -> Optional[Tuple[int, int]]:
    """Return (height, width) for an image file using MediaInfo, or None if unavailable."""
    try:
        library_path = get_media_info_library_path()
        if library_path and os.path.exists(library_path):
            media_info = MediaInfo.parse(file_path, library_file=library_path)
        else:
            media_info = MediaInfo.parse(file_path)
    except Exception:
        try:
            media_info = MediaInfo.parse(file_path)
        except Exception:
            return None

    for track in media_info.tracks:
        if getattr(track, "track_type", None) == "Image":
            height = getattr(track, "height", None)
            width = getattr(track, "width", None)
            if height is not None and width is not None:
                return int(height), int(width)
    return None


def compare_folders(folder1: str, folder2: str) -> None:
    files1 = list_files_recursive(folder1)
    files2 = list_files_recursive(folder2)

    only_in_1 = sorted(set(files1) - set(files2))
    only_in_2 = sorted(set(files2) - set(files1))

    print(f"Folder 1: {folder1}")
    print(f"Folder 2: {folder2}")
    print()
    print("Files present in Folder 1 but not Folder 2:")
    if only_in_1:
        for rel_path in only_in_1:
            print(f"  - {rel_path}")
    else:
        print("  None")
    print()
    print("Files present in Folder 2 but not Folder 1:")
    if only_in_2:
        for rel_path in only_in_2:
            print(f"  - {rel_path}")
    else:
        print("  None")
    print()

    common_files = sorted(set(files1) & set(files2))
    same_size_count = 0
    different_size_count = 0
    different_pairs: list[Tuple[str, Optional[Tuple[int, int]], Optional[Tuple[int, int]]]] = []

    for rel_path in common_files:
        path1 = files1[rel_path]
        path2 = files2[rel_path]

        dims1 = get_image_dimensions(path1)
        dims2 = get_image_dimensions(path2)

        if dims1 is None or dims2 is None:
            print(f"Skipping comparison for {rel_path}: could not read image dimensions from one or both files.")
            continue

        if dims1 == dims2:
            same_size_count += 1
        else:
            different_size_count += 1
            different_pairs.append((rel_path, dims1, dims2))

    print(f"Images with the same size: {same_size_count}")
    print(f"Images with different sizes: {different_size_count}")

    if different_pairs:
        print("\nFiles with different dimensions:")
        for rel_path, dims1, dims2 in different_pairs:
            print(f"  - {rel_path}: {dims1} vs {dims2}")
    else:
        print("\nFiles with different dimensions: None")

    total_comparable_pairs = same_size_count + different_size_count
    print(f"\nDifferent-size pair count: {different_size_count} / {total_comparable_pairs}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare two folders for file differences and image dimensions.")
    parser.add_argument("folder1", nargs="?", default=FOLDER1, help="First folder to compare")
    parser.add_argument("folder2", nargs="?", default=FOLDER2, help="Second folder to compare")
    args = parser.parse_args()

    compare_folders(args.folder1, args.folder2)

