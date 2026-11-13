#!/usr/bin/env python3

import csv
import shutil
from pathlib import Path


CSV_FILE = Path("/Volumes/LaCie/photo_matches.csv")
OUTPUT_DIR = Path("/Volumes/LaCie/recovered_correct_names")


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(CSV_FILE, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        rows = list(reader)

    print(f"Found {len(rows)} matches.")
    print()

    for row in rows:

        recovered = Path(row["recovered_file"])
        original = Path(row["original_file"])

        new_name = original.name
        destination = OUTPUT_DIR / new_name

        if destination.exists():
            raise RuntimeError(
                f"Destination already exists:\n"
                f"    {destination}\n"
                f"Refusing to overwrite it."
            )

        shutil.copy2(recovered, destination)

        print(f"{recovered.name}")
        print(f"    -> {new_name}")

    print()
    print("=" * 70)
    print(f"Created {len(rows)} correctly named files:")
    print(OUTPUT_DIR)
    print("=" * 70)


if __name__ == "__main__":
    main()