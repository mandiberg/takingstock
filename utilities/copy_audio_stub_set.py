#!/usr/bin/env python3
"""Copy WAV files for an image-ID stub set into the existing MD5 folder layout."""

import argparse
import csv
import hashlib
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


DEFAULT_STUB_CSV = "/Users/michaelmandiberg/Downloads/audio_stub_set/metasALL.csv"
DEFAULT_AUDIO_CSV = "/Volumes/OWC4/tts_office/metas_audio.csv"
DEFAULT_SOURCE_ROOT = "/Volumes/OWC4/tts_office"
DEFAULT_DESTINATION_ROOT = "/Volumes/OWC4/tts_office_stub"
DEFAULT_MANIFEST = "/Users/michaelmandiberg/Downloads/audio_stub_set/audio_stub_copy_manifest.csv"


def md5_hash_path(root: Path, filename: str) -> Path:
    digest = hashlib.md5(filename.encode("utf-8")).hexdigest()
    return root / digest[0] / digest[:2].upper() / filename


def normalize_id(value: object) -> str:
    return str(value).strip()


def read_stub_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row:
                continue
            image_id = normalize_id(row[0])
            if image_id and image_id.lower() != "image_id":
                ids.add(image_id)
    return ids


def read_matching_audio(path: Path, stub_ids: set[str]) -> dict[str, set[str]]:
    matches: dict[str, set[str]] = {}
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"image_id", "filename"}
        if reader.fieldnames is None:
            raise ValueError(f"Audio CSV has no header: {path}")
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Audio CSV is missing columns {sorted(missing)}: {path}")

        for row in reader:
            image_id = normalize_id(row["image_id"])
            if image_id not in stub_ids:
                continue
            filename = str(row["filename"]).strip()
            if not filename:
                continue
            matches.setdefault(filename, set()).add(image_id)
    return matches


def write_sliced_audio_csv(
    source_path: Path,
    destination_path: Path,
    stub_ids: set[str],
    successful_filenames: set[str],
) -> int:
    with source_path.open("r", newline="", encoding="utf-8-sig") as source_handle:
        reader = csv.DictReader(source_handle)
        if reader.fieldnames is None:
            raise ValueError(f"Audio CSV has no header: {source_path}")

        fieldnames = reader.fieldnames
        missing = {"image_id", "filename"} - set(fieldnames)
        if missing:
            raise ValueError(f"Audio CSV is missing columns {sorted(missing)}: {source_path}")

        destination_path.parent.mkdir(parents=True, exist_ok=True)
        rows_written = 0
        with destination_path.open("w", newline="", encoding="utf-8") as destination_handle:
            writer = csv.DictWriter(destination_handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in reader:
                image_id = normalize_id(row["image_id"])
                filename = str(row["filename"]).strip()
                if image_id in stub_ids and filename in successful_filenames:
                    writer.writerow(row)
                    rows_written += 1

    return rows_written


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_one(source: Path, destination: Path, verify_checksum: bool) -> tuple[str, str]:
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        if verify_checksum:
            source_checksum = sha256(source)
            destination_checksum = sha256(destination)
            if source_checksum != destination_checksum:
                return "error", "checksum mismatch after copy"
        elif source.stat().st_size != destination.stat().st_size:
            return "error", "size mismatch after copy"
        return "copied", ""
    except Exception as exc:
        return "error", str(exc)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stub-csv", default=DEFAULT_STUB_CSV)
    parser.add_argument("--audio-csv", default=DEFAULT_AUDIO_CSV)
    parser.add_argument("--source-root", default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--destination-root", default=DEFAULT_DESTINATION_ROOT)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--sliced-audio-csv",
        default="",
        help="Sliced metas_audio CSV to write (default: <destination-root>/metas_audio.csv)",
    )
    parser.add_argument("--limit", type=int, default=0, help="Copy only the first N unique filenames (0 = no limit)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify-checksum", action="store_true")
    args = parser.parse_args()

    stub_csv = Path(args.stub_csv)
    audio_csv = Path(args.audio_csv)
    source_root = Path(args.source_root)
    destination_root = Path(args.destination_root)
    manifest_path = Path(args.manifest)
    sliced_audio_csv = (
        Path(args.sliced_audio_csv)
        if args.sliced_audio_csv
        else destination_root / "metas_audio.csv"
    )

    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.limit < 0:
        parser.error("--limit cannot be negative")

    stub_ids = read_stub_ids(stub_csv)
    filename_to_ids = read_matching_audio(audio_csv, stub_ids)

    filenames = sorted(filename_to_ids)
    if args.limit:
        filenames = filenames[:args.limit]

    records: list[dict[str, str]] = []
    missing_files: list[str] = []
    malformed_files: list[str] = []
    collision_files: list[str] = []

    for filename in filenames:
        if "\r" in filename or "\n" in filename:
            malformed_files.append(repr(filename))
            source = ""
            destination = ""
            status = "malformed"
            detail = "filename contains a newline"
        else:
            source = md5_hash_path(source_root, filename)
            destination = md5_hash_path(destination_root, filename)

            if not source.is_file():
                missing_files.append(str(source))
                status = "missing"
                detail = "source file not found"
            elif destination.is_file():
                if source.stat().st_size == destination.stat().st_size:
                    status = "exists"
                    detail = ""
                else:
                    collision_files.append(str(destination))
                    status = "collision"
                    detail = "destination exists with a different size"
            else:
                status = "pending"
                detail = ""

        records.append({
            "image_ids": ";".join(sorted(filename_to_ids[filename], key=lambda value: (0, int(value)) if value.isdigit() else (1, value))),
            "filename": filename,
            "source_path": str(source),
            "destination_path": str(destination),
            "status": status,
            "detail": detail,
        })

    print(f"Stub IDs: {len(stub_ids):,}")
    print(f"Unique matching filenames: {len(filename_to_ids):,}")
    print(f"Selected filenames: {len(filenames):,}")
    print(f"Missing source files: {len(missing_files):,}")
    print(f"Malformed filenames: {len(malformed_files):,}")
    print(f"Destination collisions: {len(collision_files):,}")

    if malformed_files:
        print("Malformed filename examples:")
        for filename in malformed_files[:10]:
            print(f"  {filename}")

    if missing_files:
        print("Missing source examples:")
        for filename in missing_files[:10]:
            print(f"  {filename}")

    pending = [record for record in records if record["status"] == "pending"]

    if not args.dry_run:
        destination_root.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        for record in pending:
            record["status"] = "dry_run"
    elif pending:
        print(f"Copying {len(pending):,} files...")
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(copy_one, Path(record["source_path"]), Path(record["destination_path"]), args.verify_checksum): record
                for record in pending
            }
            completed = 0
            for future in as_completed(futures):
                record = futures[future]
                status, detail = future.result()
                record["status"] = status
                record["detail"] = detail
                completed += 1
                if completed % 500 == 0 or completed == len(pending):
                    print(f"  Completed {completed:,}/{len(pending):,}")

    if not args.dry_run:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["image_ids", "filename", "source_path", "destination_path", "status", "detail"])
            writer.writeheader()
            writer.writerows(records)

    statuses = {status: 0 for status in ("copied", "exists", "missing", "malformed", "collision", "dry_run", "error")}
    for record in records:
        status = record["status"]
        if status in statuses:
            statuses[status] += 1

    if not args.dry_run:
        successful_filenames = {
            record["filename"]
            for record in records
            if record["status"] in {"copied", "exists"}
        }
        sliced_rows = write_sliced_audio_csv(
            audio_csv,
            sliced_audio_csv,
            stub_ids,
            successful_filenames,
        )
        print(f"Sliced audio index: {sliced_audio_csv} ({sliced_rows:,} rows)")

    print("Status summary:")
    for status, count in statuses.items():
        print(f"  {status}: {count:,}")
    print(f"Manifest: {manifest_path}")

    return 1 if statuses["error"] else 0


if __name__ == "__main__":
    sys.exit(main())
