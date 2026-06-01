#!/usr/bin/env python3
"""Generate reusable spot-check artifacts for object placement assignments.

Outputs:
- <output-prefix>_slot_counts.csv
- <output-prefix>_samples.csv
- <output-prefix>_review.md
- <output-prefix>_gallery.html

Examples:
  conda run -n mp310 env PYTHONPATH=$PWD python3 \
    analysis/imagesdetections_debug/object_placement_audit/generate_spotcheck_gallery.py \
    --helper-table SegmentHelper_T11_Oct20_COCO_Custom_every40 \
    --class-id 110 \
    --slots mouth_object_id,left_hand_object_id,right_hand_object_id \
    --limit-per-slot 20 \
    --output-prefix mask110_spotcheck

    conda run -n mp310 env PYTHONPATH=$PWD python3 \
        analysis/imagesdetections_debug/object_placement_audit/generate_spotcheck_gallery.py \
        --helper-table SegmentHelper_T11_Oct20_COCO_Custom_every40 \
        --class-id 110 \
        --mode reassignment_audit \
        --target-slots mouth_object_id,left_hand_object_id,right_hand_object_id \
        --focus-slots waist_object_id,feet_object_id,shoulder_object_id \
        --limit-per-slot 40 \
        --output-prefix mask110_spotcheck
"""

import argparse
import html
import json
import math
import os
import sys
from datetime import datetime

import cv2
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.expanduser("~/Documents/GitHub/facemap/")
if REPO_ROOT not in sys.path:
    sys.path.insert(1, REPO_ROOT)

from mp_db_io import DataIO  # noqa: E402
from tools_clustering import ToolsClustering  # noqa: E402
from draw_all_zones import (  # noqa: E402
    draw_hand_intersection_areas,
    draw_mouth_rule_guides,
    draw_rect_zone,
    draw_shoulder_band,
    draw_axes,
    get_zone_rects,
    norm_to_px,
    resolve_local_image_path,
)


DEFAULT_HELPER_TABLE = "SegmentHelper_T11_Oct20_COCO_Custom_every40"
DEFAULT_SLOTS = ["feet_object_id", "left_hand_object_id", "right_hand_object_id"]
DEFAULT_TARGET_SLOTS = ["mouth_object_id", "left_hand_object_id", "right_hand_object_id"]
DEFAULT_FOCUS_SLOTS = ["waist_object_id", "feet_object_id", "shoulder_object_id"]
DEFAULT_OUTPUT_DIR = os.path.join(
    "analysis", "imagesdetections_debug", "object_placement_audit"
)
ALL_SLOT_COLS = [
    "left_hand_object_id",
    "right_hand_object_id",
    "top_face_object_id",
    "left_eye_object_id",
    "right_eye_object_id",
    "mouth_object_id",
    "shoulder_object_id",
    "waist_object_id",
    "feet_object_id",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Generate object-placement spotcheck artifacts.")
    parser.add_argument("--helper-table", default=DEFAULT_HELPER_TABLE, type=str)
    parser.add_argument("--class-id", required=True, type=int)
    parser.add_argument(
        "--mode",
        choices=["slot_assignments", "reassignment_audit"],
        default="slot_assignments",
        help="slot_assignments = sample rows assigned to requested slots; reassignment_audit = categorize detections by assignment outcome.",
    )
    parser.add_argument(
        "--slots",
        default=",".join(DEFAULT_SLOTS),
        type=str,
        help="Comma-separated slot columns from ImagesDetections.",
    )
    parser.add_argument("--limit-per-slot", default=20, type=int)
    parser.add_argument(
        "--target-slots",
        default=",".join(DEFAULT_TARGET_SLOTS),
        type=str,
        help="Comma-separated target slot columns for reassignment_audit mode.",
    )
    parser.add_argument(
        "--focus-slots",
        default=",".join(DEFAULT_FOCUS_SLOTS),
        type=str,
        help="Comma-separated non-target slot columns to track explicitly in reassignment_audit mode.",
    )
    parser.add_argument(
        "--order-mode",
        choices=["latest", "random"],
        default="latest",
        help="latest = highest detection_id first; random = ORDER BY RAND().",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, type=str)
    parser.add_argument(
        "--output-prefix",
        default=None,
        type=str,
        help="If omitted, defaults to class<id>_spotcheck.",
    )
    parser.add_argument(
        "--draw-zones",
        action="store_true",
        help="Draw zone overlays and bbox_norm boxes on local images for sampled rows.",
    )
    parser.add_argument(
        "--zone-alpha",
        type=float,
        default=0.22,
        help="Overlay alpha for zone fills when --draw-zones is enabled.",
    )
    return parser.parse_args()


def get_engine():
    io = DataIO()
    db = io.db
    return create_engine(
        "mysql+pymysql://{user}:{pw}@/{db}?unix_socket={socket}".format(
            user=db["user"],
            pw=db["pass"],
            db=db["name"],
            socket=db["unix_socket"],
        ),
        poolclass=NullPool,
    )


def sanitize_slots(raw_slots):
    slots = [s.strip() for s in raw_slots.split(",") if s.strip()]
    for slot in slots:
        # Minimal defensive check for SQL identifier shape.
        if not slot.replace("_", "").isalnum():
            raise ValueError(f"Unsafe slot identifier: {slot}")
    return slots


def validate_slot_columns(slots):
    unknown = sorted(set(slots) - set(ALL_SLOT_COLS))
    if unknown:
        raise ValueError(f"Unknown slot column(s): {', '.join(unknown)}")


def fetch_class_name(conn, class_id):
    row = conn.execute(
        text("SELECT class_name FROM YoloClasses WHERE class_id = :cid LIMIT 1"),
        {"cid": class_id},
    ).fetchone()
    if not row:
        return None
    return row[0]


def build_order_clause(order_mode, slot_col):
    if order_mode == "random":
        return "ORDER BY RAND()"
    return f"ORDER BY idt.{slot_col} DESC"


def fetch_slot_counts_and_samples(conn, helper_table, class_id, slots, limit_per_slot, order_mode):
    counts = []
    all_rows = []

    for slot_col in slots:
        q_count = text(
            f"""
            SELECT COUNT(*)
            FROM ImagesDetections idt
            JOIN {helper_table} sh ON sh.image_id = idt.image_id
            JOIN Detections d ON d.detection_id = idt.{slot_col}
            WHERE d.class_id = :cid
            """
        )
        cnt = int(conn.execute(q_count, {"cid": class_id}).scalar() or 0)
        counts.append({"slot": slot_col, "count": cnt})

        if cnt == 0:
            continue

        order_clause = build_order_clause(order_mode, slot_col)
        q_sample = text(
            f"""
            SELECT
                idt.image_id,
                :slot AS slot,
                idt.{slot_col} AS detection_id,
                d.conf,
                d.bbox_norm,
                e.nose_pixel_x,
                e.nose_pixel_y,
                e.face_height,
                i.imagename,
                i.site_name_id,
                i.contentUrl
            FROM ImagesDetections idt
            JOIN {helper_table} sh ON sh.image_id = idt.image_id
            JOIN Detections d ON d.detection_id = idt.{slot_col}
            LEFT JOIN Encodings e ON e.image_id = idt.image_id
            LEFT JOIN Images i ON i.image_id = idt.image_id
            WHERE d.class_id = :cid
            {order_clause}
            LIMIT :n
            """
        )
        rows = conn.execute(
            q_sample,
            {"cid": class_id, "slot": slot_col, "n": int(limit_per_slot)},
        ).fetchall()
        all_rows.extend([dict(r._mapping) for r in rows])

    return pd.DataFrame(counts), pd.DataFrame(all_rows)


def sort_rows(df, order_mode):
    if df.empty:
        return df
    if order_mode == "random":
        return df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    return df.sort_values(by=["detection_id"], ascending=False).reset_index(drop=True)


def fetch_reassignment_audit(conn, helper_table, class_id, target_slots, focus_slots, limit_per_bucket, order_mode):
    slot_union = "\nUNION ALL\n".join(
        [
            (
                f"SELECT image_id, {slot_col} AS detection_id, '{slot_col}' AS slot_col "
                f"FROM ImagesDetections WHERE {slot_col} IS NOT NULL"
            )
            for slot_col in ALL_SLOT_COLS
        ]
    )

    q = text(
        f"""
        WITH det AS (
            SELECT
                d.detection_id,
                d.image_id,
                d.conf,
                d.bbox_norm,
                e.nose_pixel_x,
                e.nose_pixel_y,
                e.face_height
            FROM Detections d
            JOIN {helper_table} sh ON sh.image_id = d.image_id
            LEFT JOIN Encodings e ON e.image_id = d.image_id
            WHERE d.class_id = :cid
        ),
        assignments AS (
            {slot_union}
        )
        SELECT
            det.image_id,
            det.detection_id,
            det.conf,
            det.bbox_norm,
            det.nose_pixel_x,
            det.nose_pixel_y,
            det.face_height,
            i.imagename,
            i.site_name_id,
            i.contentUrl,
            GROUP_CONCAT(DISTINCT a.slot_col ORDER BY a.slot_col SEPARATOR ';') AS assigned_slots
        FROM det
        LEFT JOIN assignments a ON a.detection_id = det.detection_id
        LEFT JOIN Images i ON i.image_id = det.image_id
        GROUP BY
            det.image_id,
            det.detection_id,
            det.conf,
            det.bbox_norm,
            det.nose_pixel_x,
            det.nose_pixel_y,
            det.face_height,
            i.imagename,
            i.site_name_id,
            i.contentUrl
        """
    )
    rows = conn.execute(q, {"cid": class_id}).fetchall()
    df = pd.DataFrame([dict(r._mapping) for r in rows])

    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    target_set = set(target_slots)
    focus_set = set(focus_slots)

    def classify(slot_str):
        if not slot_str:
            return "unassigned"
        slots = set(slot_str.split(";"))
        if slots & target_set:
            return "assigned_target_slots"
        if slots & focus_set:
            return "assigned_focus_slots"
        return "assigned_other_slots"

    df["category"] = df["assigned_slots"].apply(classify)

    counts_df = (
        df.groupby("category", dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(by=["count", "category"], ascending=[False, True])
    )

    bucket_rows = []
    for category in [
        "unassigned",
        "assigned_focus_slots",
        "assigned_other_slots",
        "assigned_target_slots",
    ]:
        cat_df = df[df["category"] == category].copy()
        if cat_df.empty:
            continue
        cat_df = sort_rows(cat_df, order_mode).head(int(limit_per_bucket))
        bucket_rows.append(cat_df)

    sample_df = pd.concat(bucket_rows, ignore_index=True) if bucket_rows else pd.DataFrame()
    return counts_df, sample_df


def render_markdown(samples_df, class_id, class_name, helper_table, slots, mode):
    title_name = class_name or "unknown"
    lines = [
        f"# Spotcheck Review: class {class_id} ({title_name})",
        "",
        f"- helper_table: {helper_table}",
        f"- generated_at_utc: {datetime.utcnow().isoformat()}Z",
        "",
    ]

    if mode == "reassignment_audit":
        for category in [
            "unassigned",
            "assigned_focus_slots",
            "assigned_other_slots",
            "assigned_target_slots",
        ]:
            lines.append(f"## {category}")
            slot_df = samples_df[samples_df["category"] == category].copy()
            if slot_df.empty:
                lines.append("")
                lines.append("No rows.")
                lines.append("")
                continue

            lines.append("")
            lines.append("| image_id | detection_id | conf | assigned_slots | overlay | imagename | contentUrl | bbox_norm |")
            lines.append("|---:|---:|---:|---|---|---|---|---|")
            for _, row in slot_df.iterrows():
                url = row.get("contentUrl") or ""
                url_md = f"[link]({url})" if url else ""
                overlay = row.get("overlay_path") or ""
                overlay_md = f"[overlay]({overlay})" if overlay else ""
                bbox = str(row.get("bbox_norm", "")).replace("|", "\\|")
                lines.append(
                    "| {image_id} | {detection_id} | {conf:.2f} | {assigned_slots} | {overlay} | {imagename} | {url} | {bbox} |".format(
                        image_id=row.get("image_id", ""),
                        detection_id=row.get("detection_id", ""),
                        conf=float(row.get("conf") or 0.0),
                        assigned_slots=(row.get("assigned_slots") or ""),
                        overlay=overlay_md,
                        imagename=(row.get("imagename") or ""),
                        url=url_md,
                        bbox=bbox,
                    )
                )
            lines.append("")
        return "\n".join(lines)

    for slot in slots:
        lines.append(f"## {slot}")
        slot_df = samples_df[samples_df["slot"] == slot].copy()
        if slot_df.empty:
            lines.append("")
            lines.append("No rows.")
            lines.append("")
            continue

        lines.append("")
        lines.append("| image_id | detection_id | conf | overlay | imagename | contentUrl | bbox_norm |")
        lines.append("|---:|---:|---:|---|---|---|---|")
        for _, row in slot_df.iterrows():
            url = row.get("contentUrl") or ""
            url_md = f"[link]({url})" if url else ""
            overlay = row.get("overlay_path") or ""
            overlay_md = f"[overlay]({overlay})" if overlay else ""
            bbox = str(row.get("bbox_norm", "")).replace("|", "\\|")
            lines.append(
                "| {image_id} | {detection_id} | {conf:.2f} | {overlay} | {imagename} | {url} | {bbox} |".format(
                    image_id=row.get("image_id", ""),
                    detection_id=row.get("detection_id", ""),
                    conf=float(row.get("conf") or 0.0),
                    overlay=overlay_md,
                    imagename=(row.get("imagename") or ""),
                    url=url_md,
                    bbox=bbox,
                )
            )
        lines.append("")

    return "\n".join(lines)


def render_html(samples_df, class_id, class_name, helper_table, slots, mode):
    title_name = class_name or "unknown"
    parts = []
    parts.append("<html><head><meta charset='utf-8'>")
    parts.append("<style>")
    parts.append("body { font-family: -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif; margin: 18px; }")
    parts.append("table { border-collapse: collapse; width: 100%; border: 1px solid #ddd; margin-bottom: 24px; }")
    parts.append("th, td { border: 1px solid #ddd; padding: 8px; text-align: left; vertical-align: top; }")
    parts.append("th { background-color: #f2f2f2; position: sticky; top: 0; }")
    parts.append("img { max-width: 220px; max-height: 220px; object-fit: contain; }")
    parts.append("code { white-space: pre-wrap; }")
    parts.append("</style></head><body>")
    parts.append(
        f"<h1>Spotcheck Gallery: class {class_id} ({html.escape(title_name)})</h1>"
    )
    parts.append(f"<p><strong>helper_table:</strong> {html.escape(helper_table)}</p>")
    parts.append(f"<p><strong>generated_at_utc:</strong> {datetime.utcnow().isoformat()}Z</p>")

    if mode == "reassignment_audit":
        for category in [
            "unassigned",
            "assigned_focus_slots",
            "assigned_other_slots",
            "assigned_target_slots",
        ]:
            parts.append(f"<h2>{html.escape(category)}</h2>")
            slot_df = samples_df[samples_df["category"] == category]
            if slot_df.empty:
                parts.append("<p>No rows.</p>")
                continue

            parts.append("<table>")
            parts.append(
                "<thead><tr><th>image_id</th><th>detection_id</th><th>conf</th><th>assigned_slots</th><th>overlay</th><th>thumbnail</th><th>open link</th><th>bbox_norm</th></tr></thead><tbody>"
            )
            for _, row in slot_df.iterrows():
                image_id = row.get("image_id", "")
                detection_id = row.get("detection_id", "")
                conf = float(row.get("conf") or 0.0)
                assigned_slots = html.escape(str(row.get("assigned_slots") or ""))
                url = (row.get("contentUrl") or "").strip()
                overlay = (row.get("overlay_path") or "").strip()
                bbox = html.escape(str(row.get("bbox_norm", "")))
                if overlay:
                    overlay_html = f"<img src='{html.escape(overlay)}' width='220'>"
                else:
                    overlay_html = ""
                if url:
                    thumb = f"<img src='{html.escape(url)}' width='220'>"
                    open_link = f"<a href='{html.escape(url)}' target='_blank'>Link</a>"
                else:
                    thumb = ""
                    open_link = ""
                parts.append(
                    f"<tr><td>{image_id}</td><td>{detection_id}</td><td>{conf:.2f}</td><td>{assigned_slots}</td><td>{overlay_html}</td><td>{thumb}</td><td>{open_link}</td><td><code>{bbox}</code></td></tr>"
                )
            parts.append("</tbody></table>")
        parts.append("</body></html>")
        return "".join(parts)

    for slot in slots:
        parts.append(f"<h2>{html.escape(slot)}</h2>")
        slot_df = samples_df[samples_df["slot"] == slot]
        if slot_df.empty:
            parts.append("<p>No rows.</p>")
            continue

        parts.append("<table>")
        parts.append(
            "<thead><tr><th>image_id</th><th>detection_id</th><th>conf</th><th>overlay</th><th>thumbnail</th><th>open link</th><th>bbox_norm</th></tr></thead><tbody>"
        )
        for _, row in slot_df.iterrows():
            image_id = row.get("image_id", "")
            detection_id = row.get("detection_id", "")
            conf = float(row.get("conf") or 0.0)
            url = (row.get("contentUrl") or "").strip()
            overlay = (row.get("overlay_path") or "").strip()
            bbox = html.escape(str(row.get("bbox_norm", "")))
            if overlay:
                overlay_html = f"<img src='{html.escape(overlay)}' width='220'>"
            else:
                overlay_html = ""
            if url:
                thumb = f"<img src='{html.escape(url)}' width='220'>"
                open_link = f"<a href='{html.escape(url)}' target='_blank'>Link</a>"
            else:
                thumb = ""
                open_link = ""
            parts.append(
                f"<tr><td>{image_id}</td><td>{detection_id}</td><td>{conf:.2f}</td><td>{overlay_html}</td><td>{thumb}</td><td>{open_link}</td><td><code>{bbox}</code></td></tr>"
            )
        parts.append("</tbody></table>")

    parts.append("</body></html>")
    return "".join(parts)


def _parse_bbox_norm(val):
    if val is None:
        return None
    if isinstance(val, dict):
        return val
    if isinstance(val, str):
        s = val.strip()
        if not s or s.lower() == "null":
            return None
        try:
            obj = json.loads(s)
            if isinstance(obj, str):
                obj = json.loads(obj)
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None
    return None


def _draw_bbox_norm_heavy(img, bbox_norm, nose_x, nose_y, face_h):
    if not bbox_norm:
        return
    required = {"left", "right", "top", "bottom"}
    if not required.issubset(set(bbox_norm.keys())):
        return

    x1, y1 = norm_to_px(float(bbox_norm["left"]), float(bbox_norm["top"]), nose_x, nose_y, face_h)
    x2, y2 = norm_to_px(float(bbox_norm["right"]), float(bbox_norm["bottom"]), nose_x, nose_y, face_h)
    left, right = min(x1, x2), max(x1, x2)
    top, bottom = min(y1, y2), max(y1, y2)

    cv2.rectangle(img, (left, top), (right, bottom), (255, 255, 255), 8)
    cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 0), 5)


def _draw_mask_threshold_boundaries(img, tc, nose_x, nose_y, face_h):
    h, w = img.shape[:2]

    boundaries = [
        ("mask_top_min", float(tc.COVID_MASK_TOP_MIN), (40, 140, 255)),
        ("mask_top_max", float(tc.COVID_MASK_TOP_MAX), (40, 140, 255)),
        ("mask_bottom_min", float(tc.COVID_MASK_BOTTOM_MIN), (255, 210, 50)),
        ("mask_bottom_max", float(tc.COVID_MASK_BOTTOM_MAX), (255, 210, 50)),
    ]

    for label, y_norm, color in boundaries:
        _, y_px = norm_to_px(0.0, y_norm, nose_x, nose_y, face_h)
        y_px = max(0, min(h - 1, int(y_px)))

        cv2.line(img, (0, y_px), (w - 1, y_px), (255, 255, 255), 4)
        cv2.line(img, (0, y_px), (w - 1, y_px), color, 2)

        cv2.putText(
            img,
            f"{label}: {y_norm:.2f}",
            (12, max(20, y_px - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            f"{label}: {y_norm:.2f}",
            (12, max(20, y_px - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            1,
            cv2.LINE_AA,
        )


def add_zone_overlays(samples_df, output_dir, class_id, mode, alpha):
    if samples_df.empty:
        samples_df["overlay_path"] = ""
        return samples_df

    io_obj = DataIO()
    tc = ToolsClustering(CLUSTER_TYPE="ObjectFusion", VERBOSE=False, session=None)
    zones = get_zone_rects(tc)
    zone_palette = {
        "left_eye_zone": (255, 120, 60),
        "right_eye_zone": (60, 180, 255),
        "under_eye_zone": (255, 200, 40),
        "eye_cover_zone": (180, 80, 255),
        "forehead_zone": (80, 220, 120),
        "waist_zone": (70, 70, 255),
        "feet_zone": (40, 180, 80),
    }

    overlay_dir = os.path.join(output_dir, "overlays", f"class_{int(class_id)}")
    os.makedirs(overlay_dir, exist_ok=True)

    overlay_paths = []
    for _, row in samples_df.iterrows():
        image_id = int(row.get("image_id"))
        detection_id = int(row.get("detection_id"))
        slot_or_cat = str(row.get("slot") or row.get("category") or "sample")
        safe_tag = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in slot_or_cat)

        imagename = row.get("imagename")
        site_name_id = row.get("site_name_id")
        image_path = resolve_local_image_path(io_obj, site_name_id, imagename)
        if not image_path:
            overlay_paths.append("")
            continue

        img = cv2.imread(image_path)
        if img is None:
            overlay_paths.append("")
            continue

        nose_x = row.get("nose_pixel_x")
        nose_y = row.get("nose_pixel_y")
        face_h = row.get("face_height")
        if nose_x is None or nose_y is None or face_h is None:
            overlay_paths.append("")
            continue

        try:
            nose_x = float(nose_x)
            nose_y = float(nose_y)
            face_h = float(face_h)
        except (TypeError, ValueError):
            overlay_paths.append("")
            continue

        if not (math.isfinite(nose_x) and math.isfinite(nose_y) and math.isfinite(face_h)):
            overlay_paths.append("")
            continue
        if face_h <= 0:
            overlay_paths.append("")
            continue

        draw_axes(img, nose_x, nose_y, face_h)
        draw_mouth_rule_guides(img, tc, nose_x, nose_y, face_h)
        for zone_name, rect in zones.items():
            draw_rect_zone(
                img,
                zone_name,
                rect,
                zone_palette.get(zone_name, (120, 120, 120)),
                alpha,
                nose_x,
                nose_y,
                face_h,
            )

        if int(class_id) == 110:
            _draw_mask_threshold_boundaries(img, tc, nose_x, nose_y, face_h)

        body_landmarks_normalized = None
        try:
            enc = io_obj.get_encodings_mongo(image_id)
            if len(enc) > 3:
                raw_nlms = enc[3]
                body_landmarks_normalized = io_obj.unpickle_array(raw_nlms) if raw_nlms is not None else None
        except Exception:
            body_landmarks_normalized = None

        if body_landmarks_normalized is not None:
            draw_shoulder_band(img, tc, body_landmarks_normalized, nose_x, nose_y, face_h, alpha)
            draw_hand_intersection_areas(img, tc, body_landmarks_normalized, nose_x, nose_y, face_h, alpha)

        bbox_norm = _parse_bbox_norm(row.get("bbox_norm"))
        _draw_bbox_norm_heavy(img, bbox_norm, nose_x, nose_y, face_h)

        out_name = f"{mode}_{safe_tag}_img{image_id}_det{detection_id}.jpg"
        out_path = os.path.join(overlay_dir, out_name)
        cv2.imwrite(out_path, img)
        rel_path = os.path.relpath(out_path, output_dir)
        overlay_paths.append(rel_path)

    out_df = samples_df.copy()
    out_df["overlay_path"] = overlay_paths
    return out_df


def main():
    args = parse_args()
    slots = sanitize_slots(args.slots)
    target_slots = sanitize_slots(args.target_slots)
    focus_slots = sanitize_slots(args.focus_slots)
    validate_slot_columns(slots)
    validate_slot_columns(target_slots)
    validate_slot_columns(focus_slots)

    output_prefix = args.output_prefix or f"class{args.class_id}_spotcheck"
    os.makedirs(args.output_dir, exist_ok=True)

    counts_path = os.path.join(args.output_dir, f"{output_prefix}_slot_counts.csv")
    samples_path = os.path.join(args.output_dir, f"{output_prefix}_samples.csv")
    review_path = os.path.join(args.output_dir, f"{output_prefix}_review.md")
    gallery_path = os.path.join(args.output_dir, f"{output_prefix}_gallery.html")

    engine = get_engine()
    with engine.connect() as conn:
        class_name = fetch_class_name(conn, args.class_id)
        if args.mode == "reassignment_audit":
            counts_df, samples_df = fetch_reassignment_audit(
                conn,
                args.helper_table,
                args.class_id,
                target_slots,
                focus_slots,
                args.limit_per_slot,
                args.order_mode,
            )
        else:
            counts_df, samples_df = fetch_slot_counts_and_samples(
                conn,
                args.helper_table,
                args.class_id,
                slots,
                args.limit_per_slot,
                args.order_mode,
            )

    if args.draw_zones:
        samples_df = add_zone_overlays(
            samples_df=samples_df,
            output_dir=args.output_dir,
            class_id=args.class_id,
            mode=args.mode,
            alpha=float(args.zone_alpha),
        )

    counts_df.to_csv(counts_path, index=False)
    samples_df.to_csv(samples_path, index=False)

    md_text = render_markdown(
        samples_df,
        args.class_id,
        class_name,
        args.helper_table,
        slots,
        args.mode,
    )
    with open(review_path, "w", encoding="utf-8") as f:
        f.write(md_text)

    html_text = render_html(
        samples_df,
        args.class_id,
        class_name,
        args.helper_table,
        slots,
        args.mode,
    )
    with open(gallery_path, "w", encoding="utf-8") as f:
        f.write(html_text)

    print("Done.")
    print("Counts:")
    if counts_df.empty:
        print("  (no rows)")
    else:
        name_col = "slot" if "slot" in counts_df.columns else "category"
        for _, row in counts_df.iterrows():
            print(f"  {row[name_col]}: {int(row['count'])}")
    print(f"Wrote: {counts_path}")
    print(f"Wrote: {samples_path}")
    print(f"Wrote: {review_path}")
    print(f"Wrote: {gallery_path}")


if __name__ == "__main__":
    main()
