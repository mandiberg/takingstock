import argparse
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock


DEFAULT_SOURCE = "/Volumes/OWC5/cache_folder"
DEFAULT_DEST = "/Volumes/LaCie/cache_folder"
OUTPUT_INTERVAL = 500


def parse_args():
	parser = argparse.ArgumentParser(
		description=(
			"Copy cache folders/files from source to destination without overwriting. "
			"If a folder is missing at destination, copy the folder; if it exists, copy only missing .jpg files."
		)
	)
	parser.add_argument("--source", default=DEFAULT_SOURCE, help="Source cache_folder path")
	parser.add_argument("--dest", default=DEFAULT_DEST, help="Destination cache_folder path")
	parser.add_argument(
		"--workers",
		type=int,
		default=max(4, (os.cpu_count() or 8)),
		help="Number of worker threads",
	)
	parser.add_argument(
		"--max-in-flight",
		type=int,
		default=0,
		help="Max queued futures before draining (0 = workers * 4)",
	)
	parser.add_argument(
		"--dry-run",
		action="store_true",
		help="Preview actions without copying files",
	)
	parser.add_argument(
		"--verbose",
		action="store_true",
		help="Print per-folder diagnostics",
	)
	return parser.parse_args()


def iter_top_level_folders(root_path):
	with os.scandir(root_path) as entries:
		for entry in entries:
			if entry.is_dir():
				yield entry.name


def count_top_level_jpgs(folder_path):
	count = 0
	with os.scandir(folder_path) as entries:
		for entry in entries:
			if entry.is_file() and entry.name.lower().endswith(".jpg"):
				count += 1
	return count


def ignore_non_jpg(_dir, names):
	ignored = []
	for name in names:
		if not name.lower().endswith(".jpg"):
			ignored.append(name)
	return ignored


def process_one_folder(folder_name, source_root, dest_root, dry_run=False, verbose=False):
	src_folder = os.path.join(source_root, folder_name)
	dst_folder = os.path.join(dest_root, folder_name)

	result = {
		"folders_processed": 1,
		"folders_copied": 0,
		"folders_existing": 0,
		"files_copied": 0,
		"files_existing": 0,
		"files_non_jpg_skipped": 0,
		"errors": 0,
	}

	if not os.path.isdir(src_folder):
		result["errors"] += 1
		return result

	try:
		if not os.path.exists(dst_folder):
			jpg_count = count_top_level_jpgs(src_folder)
			if verbose:
				print(f"[folder-missing] {folder_name} -> copy entire folder ({jpg_count} jpg)")

			if not dry_run:
				shutil.copytree(src_folder, dst_folder, ignore=ignore_non_jpg)

			result["folders_copied"] += 1
			result["files_copied"] += jpg_count
			return result

		result["folders_existing"] += 1

		with os.scandir(src_folder) as src_entries:
			for src_entry in src_entries:
				if not src_entry.is_file():
					continue

				if not src_entry.name.lower().endswith(".jpg"):
					result["files_non_jpg_skipped"] += 1
					continue

				dst_file = os.path.join(dst_folder, src_entry.name)
				if os.path.exists(dst_file):
					result["files_existing"] += 1
					continue

				if verbose:
					print(f"[copy-file] {src_entry.path} -> {dst_file}")

				if not dry_run:
					shutil.copy2(src_entry.path, dst_file)

				result["files_copied"] += 1

	except Exception as err:
		result["errors"] += 1
		print(f"[error] folder={folder_name} reason={err}")

	return result


def merge_counts(total, partial):
	for key in total:
		total[key] += partial.get(key, 0)


def main():
	args = parse_args()

	source_root = os.path.abspath(args.source)
	dest_root = os.path.abspath(args.dest)
	workers = max(1, int(args.workers))
	max_in_flight = int(args.max_in_flight) if args.max_in_flight and args.max_in_flight > 0 else workers * 4

	if not os.path.isdir(source_root):
		raise FileNotFoundError(f"Source folder does not exist: {source_root}")

	Path(dest_root).mkdir(parents=True, exist_ok=True)

	print("Starting cache copy")
	print(f"source: {source_root}")
	print(f"dest:   {dest_root}")
	print(f"mode:   {'DRY RUN' if args.dry_run else 'EXECUTE'}")
	print(f"workers: {workers}, max_in_flight: {max_in_flight}")

	totals = {
		"folders_processed": 0,
		"folders_copied": 0,
		"folders_existing": 0,
		"files_copied": 0,
		"files_existing": 0,
		"files_non_jpg_skipped": 0,
		"errors": 0,
	}

	lock = Lock()
	interval_count = 0
	interval_start = time.time()
	all_start = interval_start

	def handle_result(partial):
		nonlocal interval_count, interval_start
		with lock:
			merge_counts(totals, partial)
			interval_count += partial.get("folders_processed", 0)

			if interval_count >= OUTPUT_INTERVAL:
				elapsed = time.time() - interval_start
				print(
					f"Interval folders={interval_count} time={elapsed:.1f}s "
					f"folders_copied={totals['folders_copied']} folders_existing={totals['folders_existing']} "
					f"files_copied={totals['files_copied']} files_existing={totals['files_existing']} "
					f"errors={totals['errors']}"
				)
				interval_count = 0
				interval_start = time.time()

	submitted = 0
	futures = []

	with ThreadPoolExecutor(max_workers=workers) as executor:
		for folder_name in iter_top_level_folders(source_root):
			futures.append(
				executor.submit(
					process_one_folder,
					folder_name,
					source_root,
					dest_root,
					args.dry_run,
					args.verbose,
				)
			)
			submitted += 1

			if len(futures) >= max_in_flight:
				for future in as_completed(futures):
					handle_result(future.result())
				futures.clear()

		for future in as_completed(futures):
			handle_result(future.result())

	total_elapsed = time.time() - all_start
	print("Done")
	print(f"folders_submitted={submitted}")
	print(
		"summary: "
		f"folders_processed={totals['folders_processed']}, "
		f"folders_copied={totals['folders_copied']}, "
		f"folders_existing={totals['folders_existing']}, "
		f"files_copied={totals['files_copied']}, "
		f"files_existing={totals['files_existing']}, "
		f"files_non_jpg_skipped={totals['files_non_jpg_skipped']}, "
		f"errors={totals['errors']}, "
		f"elapsed={total_elapsed:.1f}s"
	)


if __name__ == "__main__":
	main()
