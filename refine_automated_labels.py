#!/usr/bin/env python3
"""Refine automated YOLO labels from surviving debug images.

This script walks a main training-data folder, finds each pair of
`review_new_detections` and `debug` folders, builds a manifest of the
remaining debug images and their surviving class ids, and writes a
`review_refined_detections` branch containing only the surviving images
and a collapsed label file per image.

Rules implemented:
- Existing labels always win over new labels.
- New labels are defined by the row-aware rule supplied by the user.
- New labels are filtered by surviving debug class ids.
- Overlapping new labels are collapsed using IoU and priority rules.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
from typing import DefaultDict, Dict, Iterable, List, Optional, Sequence, Set, Tuple


MAIN_FOLDER_DEFAULT = "/Users/michaelmandiberg/Documents/YOLO_Training_Data/sorted_images_reprocess"
REVIEW_DIR_NAME = "review_new_detections"
DEBUG_DIR_NAME = "debug"
OUTPUT_DIR_NAME = "review_refined_detections"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# The user asked for 0.10 to stay consistent with the protected-conflict logic.
IOU_COLLAPSE_THRESHOLD = 0.10

# Priority ladder for new detections. Larger numbers win.
# Specific pairwise rules requested by the user are encoded here.
NEW_CLASS_PRIORITY = {
	132: 700,
	133: 650,
	137: 600,
	123: 500,
	127: 450,
	136: 400,
	93: 350,
	135: 100,
}


@dataclass(frozen=True)
class Annotation:
	row_index: int
	class_id: int
	x_center: float
	y_center: float
	width: float
	height: float
	raw_line: str
	is_new: bool

	def bbox_xyxy(self) -> Dict[str, float]:
		left = self.x_center - (self.width / 2.0)
		top = self.y_center - (self.height / 2.0)
		right = self.x_center + (self.width / 2.0)
		bottom = self.y_center + (self.height / 2.0)
		return {"left": left, "top": top, "right": right, "bottom": bottom}

	def to_yolo_line(self) -> str:
		return (
			f"{self.class_id} {self.x_center:.6f} {self.y_center:.6f} "
			f"{self.width:.6f} {self.height:.6f}\n"
		)


def is_image_file(filename: str) -> bool:
	return os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS


def normalize_stem(filename: str) -> str:
	return os.path.splitext(os.path.basename(filename))[0]


def parse_debug_image_id(filename: str) -> Optional[str]:
	"""Extract image_id from debug filenames like 0.87_12345_YOLO_debug.jpg."""
	stem = normalize_stem(filename)
	match = re.match(r"^(?:\d+(?:\.\d+)?_)?(.+?)_YOLO_debug$", stem)
	if match:
		return match.group(1)
	if stem.endswith("_YOLO_debug"):
		return stem[:-len("_YOLO_debug")].lstrip("0123456789._") or None
	return stem or None


def parse_label_image_id(filename: str) -> Optional[str]:
	"""Extract image_id from label filenames."""
	stem = normalize_stem(filename)
	if not stem:
		return None
	match = re.match(r"^(?:\d+(?:\.\d+)?_)?(.+?)_YOLO_debug$", stem)
	if match:
		return match.group(1)
	return stem


def parse_class_folder_name(folder_name: str) -> List[int]:
	parts = [part for part in folder_name.split("_") if part.isdigit()]
	return [int(part) for part in parts]


def is_new_detection(class_id: int, row_index: int) -> bool:
	return (class_id > 120 and class_id != 127) or (class_id in {93, 127} and row_index > 1)


def load_label_file(label_path: str) -> List[Annotation]:
	annotations: List[Annotation] = []
	with open(label_path, "r", encoding="utf-8") as handle:
		for row_index, line in enumerate(handle, start=1):
			raw_line = line.strip()
			if not raw_line:
				continue
			parts = raw_line.split()
			if len(parts) < 5:
				print(f"Warning: skipping malformed label row {row_index} in {label_path}: {raw_line}")
				continue
			try:
				class_id = int(float(parts[0]))
				x_center = float(parts[1])
				y_center = float(parts[2])
				width = float(parts[3])
				height = float(parts[4])
			except ValueError:
				print(f"Warning: skipping non-numeric label row {row_index} in {label_path}: {raw_line}")
				continue

			annotations.append(
				Annotation(
					row_index=row_index,
					class_id=class_id,
					x_center=x_center,
					y_center=y_center,
					width=width,
					height=height,
					raw_line=raw_line,
					is_new=is_new_detection(class_id, row_index),
				)
			)
	return annotations


def bbox_iou(a: Dict[str, float], b: Dict[str, float]) -> float:
	inter_left = max(a["left"], b["left"])
	inter_top = max(a["top"], b["top"])
	inter_right = min(a["right"], b["right"])
	inter_bottom = min(a["bottom"], b["bottom"])

	if inter_right <= inter_left or inter_bottom <= inter_top:
		return 0.0

	intersection = (inter_right - inter_left) * (inter_bottom - inter_top)
	area_a = max(0.0, a["right"] - a["left"]) * max(0.0, a["bottom"] - a["top"])
	area_b = max(0.0, b["right"] - b["left"]) * max(0.0, b["bottom"] - b["top"])
	union = area_a + area_b - intersection
	if union <= 0:
		return 0.0
	return intersection / union


def class_priority(class_id: int) -> int:
	return NEW_CLASS_PRIORITY.get(class_id, 0)


def load_surviving_debug_manifest(debug_dir: str) -> Tuple[Dict[str, Set[int]], Dict[str, str]]:
	"""Return image_id -> surviving class ids and image_id -> source debug image path."""
	surviving_classes: DefaultDict[str, Set[int]] = defaultdict(set)
	source_debug_image: Dict[str, str] = {}

	for root, _, files in os.walk(debug_dir):
		relative_root = os.path.relpath(root, debug_dir)
		if relative_root == ".":
			continue

		class_ids: List[int] = []
		for part in relative_root.split(os.sep):
			class_ids = parse_class_folder_name(part)
			if class_ids:
				break
		if not class_ids:
			continue

		for filename in files:
			if not is_image_file(filename):
				continue
			image_id = parse_debug_image_id(filename)
			if image_id is None:
				continue
			surviving_classes[image_id].update(class_ids)
			source_debug_image.setdefault(image_id, os.path.join(root, filename))

	return dict(surviving_classes), source_debug_image


def index_review_labels(review_dir: str) -> Dict[str, str]:
	labels_dir = os.path.join(review_dir, "labels")
	label_index: Dict[str, str] = {}

	if not os.path.isdir(labels_dir):
		return label_index

	for root, _, files in os.walk(labels_dir):
		for filename in files:
			if not filename.lower().endswith(".txt"):
				continue
			image_id = parse_label_image_id(filename)
			if image_id is None:
				continue
			label_index[image_id] = os.path.join(root, filename)

	return label_index


def index_image_sources(image_dirs: Sequence[str]) -> Dict[str, str]:
	"""Return image_id -> source image path from a prioritized list of folders."""
	image_index: Dict[str, str] = {}
	for image_dir in image_dirs:
		if not os.path.isdir(image_dir):
			continue
		for root, _, files in os.walk(image_dir):
			for filename in files:
				if not is_image_file(filename):
					continue
				image_id = parse_debug_image_id(filename)
				if image_id is None:
					continue
				# Keep the first match so earlier folders in image_dirs win.
				image_index.setdefault(image_id, os.path.join(root, filename))
	return image_index


def overlaps_any(annotation: Annotation, others: Sequence[Annotation], threshold: float) -> bool:
	annotation_box = annotation.bbox_xyxy()
	for other in others:
		if bbox_iou(annotation_box, other.bbox_xyxy()) >= threshold:
			return True
	return False


def connected_components(annotations: Sequence[Annotation], threshold: float) -> List[List[int]]:
	adjacency: List[Set[int]] = [set() for _ in annotations]
	for i in range(len(annotations)):
		bbox_i = annotations[i].bbox_xyxy()
		for j in range(i + 1, len(annotations)):
			if bbox_iou(bbox_i, annotations[j].bbox_xyxy()) >= threshold:
				adjacency[i].add(j)
				adjacency[j].add(i)

	components: List[List[int]] = []
	seen: Set[int] = set()
	for start in range(len(annotations)):
		if start in seen:
			continue
		stack = [start]
		component: List[int] = []
		seen.add(start)
		while stack:
			current = stack.pop()
			component.append(current)
			for neighbor in adjacency[current]:
				if neighbor not in seen:
					seen.add(neighbor)
					stack.append(neighbor)
		components.append(component)
	return components


def choose_new_winner(component: Sequence[Annotation]) -> Annotation:
	return max(
		component,
		key=lambda ann: (
			class_priority(ann.class_id),
			-ann.row_index,
			-ann.class_id,
		),
	)


def refine_annotations(
	annotations: Sequence[Annotation],
	surviving_classes: Set[int],
	iou_threshold: float,
) -> Tuple[List[Annotation], Dict[str, int]]:
	existing = [ann for ann in annotations if not ann.is_new]
	new_candidates = [
		ann for ann in annotations
		if ann.is_new and ann.class_id in surviving_classes
	]

	new_after_existing: List[Annotation] = []
	suppressed_by_existing = 0
	for ann in new_candidates:
		if overlaps_any(ann, existing, iou_threshold):
			suppressed_by_existing += 1
			continue
		new_after_existing.append(ann)

	winners: List[Annotation] = []
	suppressed_by_priority = 0
	if new_after_existing:
		for component_indices in connected_components(new_after_existing, iou_threshold):
			component = [new_after_existing[i] for i in component_indices]
			winner = choose_new_winner(component)
			winners.append(winner)
			suppressed_by_priority += max(0, len(component) - 1)

	winners.sort(key=lambda ann: (-class_priority(ann.class_id), ann.row_index, ann.class_id))
	existing.sort(key=lambda ann: ann.row_index)

	refined = existing + winners
	stats = {
		"existing_count": len(existing),
		"new_candidates": len(new_candidates),
		"suppressed_by_existing": suppressed_by_existing,
		"suppressed_by_priority": suppressed_by_priority,
		"refined_count": len(refined),
	}
	return refined, stats


def write_labels(label_path: str, annotations: Sequence[Annotation]) -> None:
	os.makedirs(os.path.dirname(label_path), exist_ok=True)
	with open(label_path, "w", encoding="utf-8") as handle:
		for ann in annotations:
			handle.write(ann.to_yolo_line())


def copy_image(source_path: str, target_path: str) -> None:
	os.makedirs(os.path.dirname(target_path), exist_ok=True)
	shutil.copy2(source_path, target_path)


def process_review_root(main_folder: str, review_dir: str, debug_dir: str, iou_threshold: float) -> Dict[str, object]:
	root_parent = os.path.dirname(review_dir)
	output_root = os.path.join(root_parent, OUTPUT_DIR_NAME)
	output_images_dir = os.path.join(output_root, "images")
	output_labels_dir = os.path.join(output_root, "labels")
	os.makedirs(output_images_dir, exist_ok=True)
	os.makedirs(output_labels_dir, exist_ok=True)

	surviving_classes_by_image_id, source_debug_image_by_image_id = load_surviving_debug_manifest(debug_dir)
	review_label_index = index_review_labels(review_dir)
	review_image_index = index_image_sources(
		[
			os.path.join(review_dir, "images"),
			os.path.join(root_parent, "images"),
		]
	)

	manifest: Dict[str, Dict[str, object]] = {}
	stats = {
		"images_found": len(surviving_classes_by_image_id),
		"images_written": 0,
		"labels_written": 0,
		"images_missing_labels": 0,
		"images_missing_debug_source": 0,
	}

	for image_id in sorted(surviving_classes_by_image_id.keys()):
		surviving_classes = surviving_classes_by_image_id[image_id]
		label_path = review_label_index.get(image_id)
		debug_source_path = source_debug_image_by_image_id.get(image_id)
		image_source_path = review_image_index.get(image_id)

		if label_path is None:
			print(f"[skip] {image_id}: no matching review label file in {review_dir}")
			stats["images_missing_labels"] += 1
			continue

		if image_source_path is None or not os.path.exists(image_source_path):
			print(f"[skip] {image_id}: no original review image found in {review_dir}/images or parent images")
			stats["images_missing_debug_source"] += 1
			continue

		annotations = load_label_file(label_path)
		refined_annotations, refine_stats = refine_annotations(
			annotations,
			surviving_classes,
			iou_threshold=iou_threshold,
		)

		if not refined_annotations:
			print(f"[skip] {image_id}: no refined annotations survived")
			continue

		image_ext = os.path.splitext(image_source_path)[1].lower() or ".jpg"
		output_image_path = os.path.join(output_images_dir, f"{image_id}{image_ext}")
		output_label_path = os.path.join(output_labels_dir, f"{image_id}.txt")

		copy_image(image_source_path, output_image_path)
		write_labels(output_label_path, refined_annotations)

		manifest[image_id] = {
			"debug_source": os.path.relpath(debug_source_path, main_folder),
			"label_source": os.path.relpath(label_path, main_folder),
			"surviving_classes": sorted(surviving_classes),
			"refined_label": os.path.relpath(output_label_path, main_folder),
			"refined_image": os.path.relpath(output_image_path, main_folder),
			"stats": refine_stats,
		}
		stats["images_written"] += 1
		stats["labels_written"] += 1

		print(
			f"[write] {image_id}: existing={refine_stats['existing_count']} "
			f"new_candidates={refine_stats['new_candidates']} "
			f"suppressed_existing={refine_stats['suppressed_by_existing']} "
			f"suppressed_priority={refine_stats['suppressed_by_priority']} "
			f"refined={refine_stats['refined_count']}"
		)

	manifest_path = os.path.join(output_root, "remaining_debug_images.json")
	with open(manifest_path, "w", encoding="utf-8") as handle:
		json.dump(manifest, handle, indent=2, sort_keys=True)

	print(
		f"[summary] review_dir={review_dir} debug_dir={debug_dir} "
		f"found={stats['images_found']} written={stats['images_written']} "
		f"missing_labels={stats['images_missing_labels']} missing_debug_source={stats['images_missing_debug_source']}"
	)
	print(f"[summary] refined output: {output_root}")
	print(f"[summary] manifest: {manifest_path}")

	return {
		"review_dir": review_dir,
		"debug_dir": debug_dir,
		"output_root": output_root,
		"stats": stats,
	}


def find_review_debug_pairs(main_folder: str) -> List[Tuple[str, str]]:
	pairs: List[Tuple[str, str]] = []
	for root, dirs, _ in os.walk(main_folder):
		if REVIEW_DIR_NAME in dirs and DEBUG_DIR_NAME in dirs:
			review_dir = os.path.join(root, REVIEW_DIR_NAME)
			debug_dir = os.path.join(root, DEBUG_DIR_NAME)
			pairs.append((review_dir, debug_dir))
	pairs.sort()
	return pairs


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Refine automated YOLO labels from surviving debug images.")
	parser.add_argument(
		"--main-folder",
		default=MAIN_FOLDER_DEFAULT,
		help="Root folder containing per-dataset test_output folders.",
	)
	parser.add_argument(
		"--iou-threshold",
		type=float,
		default=IOU_COLLAPSE_THRESHOLD,
		help="IoU threshold used to collapse overlapping detections.",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	main_folder = os.path.abspath(args.main_folder)

	if not os.path.isdir(main_folder):
		raise FileNotFoundError(f"MAIN_FOLDER does not exist or is not a directory: {main_folder}")

	pairs = find_review_debug_pairs(main_folder)
	if not pairs:
		print(f"No {REVIEW_DIR_NAME!r} and {DEBUG_DIR_NAME!r} folder pairs found under {main_folder}")
		return

	print(f"Found {len(pairs)} review/debug folder pair(s) under {main_folder}")
	for review_dir, debug_dir in pairs:
		process_review_root(main_folder, review_dir, debug_dir, args.iou_threshold)


if __name__ == "__main__":
	main()
