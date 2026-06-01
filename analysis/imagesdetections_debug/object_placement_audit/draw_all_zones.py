#!/usr/bin/env python3
"""Draw all rectangle-intersection zones from ToolsClustering.

The script reads zone values directly from ToolsClustering so changes in
`tools_clustering.py` are reflected automatically.

Examples:
  # Blank canvas with synthetic nose/face-height reference
  conda run -n mp310 env PYTHONPATH=$PWD python3 \
    analysis/imagesdetections_debug/object_placement_audit/draw_all_zones.py \
    --output analysis/imagesdetections_debug/object_placement_audit/zones_blank.png

  # Overlay on a real image (provide nose and face-height in pixels)
  conda run -n mp310 env PYTHONPATH=$PWD python3 \
    analysis/imagesdetections_debug/object_placement_audit/draw_all_zones.py \
    --image /path/to/image.jpg \
    --nose-x 960 --nose-y 420 --face-height 520 \
    --output analysis/imagesdetections_debug/object_placement_audit/zones_on_image.png

    conda run -n mp310 env PYTHONPATH=$PWD 
    python3 analysis/imagesdetections_debug/object_placement_audit/draw_all_zones.py 
    --image-id 128306454 --output analysis/imagesdetections_debug/object_placement_audit/zones_from_db_128306454.png

    conda run -n mp310 env PYTHONPATH=$PWD python3 analysis/imagesdetections_debug/object_placement_audit/draw_all_zones.py \
    --image-id 128306454 \
    --output analysis/imagesdetections_debug/object_placement_audit/zones_from_db_128306454.png

"""

import argparse
import os
import sys
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from cv2 import dnn_superres
from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tools_clustering import ToolsClustering  # noqa: E402
from mp_db_io import DataIO  # noqa: E402

IS_SSD=True
VERBOSE=False
SSD_PATH="/Volumes/SanDiskBlack/segment_images_82_money_cards"
# SSD_PATH = None
io_obj = DataIO(IS_SSD=IS_SSD, VERBOSE=VERBOSE, SSD_PATH=SSD_PATH)

Color = Tuple[int, int, int]
UPSCALE_MODEL_PATH = os.path.join(REPO_ROOT, "models", "FSRCNN_x4.pb")
UP_RES_4X = False
BATCH_OUTPUT_SUBDIR = "batch"

INPUT_LIST = [6301419, 82968541, 86341312, 107057552, 107989301, 110249649, 125734939]


def parse_args():
    parser = argparse.ArgumentParser(description="Draw ToolsClustering rectangle zones.")
    parser.add_argument(
        "--image-id",
        type=int,
        default=None,
        help="Image ID to fetch image path + nose/face-height from MySQL.",
    )
    parser.add_argument("--image", type=str, default=None, help="Optional image to draw on.")
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(
            "analysis",
            "imagesdetections_debug",
            "object_placement_audit",
            "zones_overlay.png",
        ),
    )
    parser.add_argument("--width", type=int, default=1600, help="Canvas width when --image is omitted.")
    parser.add_argument("--height", type=int, default=1000, help="Canvas height when --image is omitted.")
    parser.add_argument(
        "--nose-x",
        type=float,
        default=None,
        help="Nose x pixel for mapping normalized coordinates onto image/canvas.",
    )
    parser.add_argument(
        "--nose-y",
        type=float,
        default=None,
        help="Nose y pixel for mapping normalized coordinates onto image/canvas.",
    )
    parser.add_argument(
        "--face-height",
        type=float,
        default=None,
        help="Face height in pixels (scale for normalized coordinates).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.25,
        help="Fill alpha (0-1) for zone overlays.",
    )
    return parser.parse_args()


def get_engine(io_obj: DataIO):
    db = io_obj.db
    return create_engine(
        "mysql+pymysql://{user}:{pw}@/{db}?unix_socket={socket}".format(
            user=db["user"],
            pw=db["pass"],
            db=db["name"],
            socket=db["unix_socket"],
        ),
        poolclass=NullPool,
    )


def set_upscale_model(upscale_model_path: str):
    print("model_path", upscale_model_path)
    sr = dnn_superres.DnnSuperResImpl_create()
    sr.readModel(upscale_model_path)
    sr.setModel("fsrcnn", 4)
    return sr


def upscale_image_4x(img: np.ndarray, upscale_model_path: str) -> np.ndarray:
    if img is None:
        return img
    if not upscale_model_path or not os.path.exists(upscale_model_path):
        print(f"FSRCNN model not found at {upscale_model_path}, skipping 4x upscaling")
        return img
    try:
        sr = set_upscale_model(upscale_model_path)
        upsized_image = sr.upsample(img)
        if upsized_image is not None:
            print(f"Applied FSRCNN 4x upscaling: {img.shape[:2]} -> {upsized_image.shape[:2]}")
            return upsized_image
    except Exception as e:
        print(f"FSRCNN upscaling failed ({e}), continuing with original image")
    return img


def scaled_thickness(base_thickness: int, ui_scale: float) -> int:
    return max(1, int(round(base_thickness * max(ui_scale, 1.0))))


def resolve_local_image_path(
    io_obj: DataIO,
    site_name_id: Optional[int],
    imagename: Optional[str],
    content_url: Optional[str] = None,
) -> Optional[str]:
    if not imagename and not content_url:
        return None

    def filename_from_content_url(url: Optional[str]) -> Optional[str]:
        if not isinstance(url, str) or not url.strip():
            return None
        file_name_path = url.strip().split("?")[0]
        file_name = os.path.basename(file_name_path)
        if not file_name:
            return None
        lower_name = file_name.lower()
        if lower_name.endswith(".jpeg"):
            file_name = file_name[:-5] + ".jpg"
        elif not lower_name.endswith(".jpg"):
            file_name = file_name + ".jpg"
        return file_name

    def dedupe_preserve_order(values):
        seen = set()
        output = []
        for value in values:
            if not value or value in seen:
                continue
            seen.add(value)
            output.append(value)
        return output

    url_filename = filename_from_content_url(content_url)
    imagename_basename = os.path.basename(imagename) if isinstance(imagename, str) else None

    candidate_names = dedupe_preserve_order(
        [
            imagename,
            imagename_basename,
            url_filename,
        ]
    )

    candidates = []

    # If DB already stores absolute local path.
    if isinstance(imagename, str):
        candidates.append(imagename)

    # Try contentUrl directly if it already points to a local absolute path.
    if isinstance(content_url, str):
        candidates.append(content_url)

    # Try hashed folder pattern used by ingestion scripts.
    if url_filename:
        try:
            hash_folder, hash_subfolder = io_obj.get_hash_folders(url_filename)
            hashed_relpath = os.path.join(hash_folder, hash_subfolder, url_filename)
            candidates.append(hashed_relpath)
        except Exception:
            pass

    # Try folder mapping by site id.
    if site_name_id is not None:
        try:
            site_idx = int(site_name_id)
            if 0 <= site_idx < len(io_obj.folder_list):
                site_root = io_obj.folder_list[site_idx]
                if site_root:
                    for candidate_name in candidate_names:
                        candidates.append(os.path.join(site_root, candidate_name))

                    if url_filename:
                        try:
                            hash_folder, hash_subfolder = io_obj.get_hash_folders(url_filename)
                            candidates.append(
                                os.path.join(site_root, hash_folder, hash_subfolder, url_filename)
                            )
                        except Exception:
                            pass
        except Exception:
            pass

    # Try a couple of global roots as fallback.
    for root_attr in ("ROOT_PROD", "ROOT", "ROOTSSD", "ROOT18", "ROOT54"):
        root_val = getattr(io_obj, root_attr, None)
        if root_val:
            for candidate_name in candidate_names:
                candidates.append(os.path.join(root_val, candidate_name))

            if url_filename:
                try:
                    hash_folder, hash_subfolder = io_obj.get_hash_folders(url_filename)
                    candidates.append(os.path.join(root_val, hash_folder, hash_subfolder, url_filename))
                except Exception:
                    pass

    for path in candidates:
        if path and os.path.exists(path):
            return path

    return None


def fetch_image_context_from_mysql(image_id: int) -> Dict[str, Optional[float]]:
    engine = get_engine(io_obj)

    q = text(
        """
        SELECT
            i.image_id,
            i.imagename,
            i.contentUrl,
            i.site_name_id,
            e.nose_pixel_x,
            e.nose_pixel_y,
            e.face_height,
            e.encoding_id,
            e.is_face
        FROM Images i
        JOIN Encodings e ON e.image_id = i.image_id
        WHERE i.image_id = :image_id
          AND e.nose_pixel_x IS NOT NULL
          AND e.nose_pixel_y IS NOT NULL
          AND e.face_height IS NOT NULL
        ORDER BY
            COALESCE(e.is_face, 0) DESC,
            e.face_height DESC,
            e.encoding_id DESC
        LIMIT 1
        """
    )

    with engine.connect() as conn:
        row = conn.execute(q, {"image_id": int(image_id)}).fetchone()

    if not row:
        raise RuntimeError(
            f"No Images/Encodings row found with nose+face_height for image_id={image_id}."
        )

    data = dict(row._mapping)
    local_path = resolve_local_image_path(
        io_obj,
        data.get("site_name_id"),
        data.get("imagename"),
        data.get("contentUrl"),
    )
    data["resolved_local_path"] = local_path

    # Try to fetch normalized body landmarks from mongo for dynamic shoulder-band drawing.
    body_landmarks_normalized = None
    try:
        import pickle as _pickle
        import pymongo as _pymongo
        _mongo_client = _pymongo.MongoClient(io_obj.dbmongo["host"])
        _mongo_db = _mongo_client[io_obj.dbmongo["name"]]
        _col = _mongo_db["body_landmarks_norm"]
        _doc = _col.find_one({"image_id": int(image_id)})
        if _doc and _doc.get("nlms"):
            body_landmarks_normalized = _pickle.loads(_doc["nlms"])
            print(f"  mongo body_landmarks_normalized: {type(body_landmarks_normalized)}")
        else:
            print(f"  mongo: no body_landmarks_norm doc for image_id={image_id}")
    except Exception as e:
        print(f"  mongo fetch failed: {e}")
        body_landmarks_normalized = None

    data["body_landmarks_normalized"] = body_landmarks_normalized
    return data


def get_zone_rects(tc: ToolsClustering) -> Dict[str, Dict[str, float]]:
    # All of these are rectangle-intersection zones, sourced from ToolsClustering attributes.
    return {
        "left_eye_zone": {
            "x_min": tc.LEFT_EYE_X_MIN,
            "x_max": tc.LEFT_EYE_X_MAX,
            "y_top": tc.EYE_ZONE_TOP,
            "y_bottom": tc.EYE_ZONE_BOTTOM,
        },
        "right_eye_zone": {
            "x_min": tc.RIGHT_EYE_X_MIN,
            "x_max": tc.RIGHT_EYE_X_MAX,
            "y_top": tc.EYE_ZONE_TOP,
            "y_bottom": tc.EYE_ZONE_BOTTOM,
        },
        "under_eye_zone": {
            "x_min": tc.LEFT_EYE_X_MIN,
            "x_max": tc.RIGHT_EYE_X_MAX,
            "y_top": tc.UNDER_EYE_ZONE_TOP,
            "y_bottom": tc.UNDER_EYE_ZONE_BOTTOM,
        },
        "eye_cover_zone": {
            "x_min": tc.LEFT_EYE_X_MIN,
            "x_max": tc.RIGHT_EYE_X_MAX,
            "y_top": tc.EYE_COVER_ZONE_TOP,
            "y_bottom": tc.EYE_COVER_ZONE_BOTTOM,
        },
        "forehead_zone": {
            "x_min": tc.FOREHEAD_X_MIN,
            "x_max": tc.FOREHEAD_X_MAX,
            "y_top": tc.FOREHEAD_ZONE_TOP,
            "y_bottom": tc.FOREHEAD_ZONE_BOTTOM,
        },
        "waist_zone": {
            "x_min": tc.WAIST_X_MIN,
            "x_max": tc.WAIST_X_MAX,
            "y_top": tc.WAIST_ZONE_TOP,
            "y_bottom": tc.WAIST_ZONE_BOTTOM,
        },
        "feet_zone": {
            "x_min": tc.FEET_X_MIN,
            "x_max": tc.FEET_X_MAX,
            "y_top": tc.FEET_ZONE_TOP,
            "y_bottom": tc.FEET_ZONE_BOTTOM,
        },
    }


def norm_to_px(x_norm: float, y_norm: float, nose_x: float, nose_y: float, face_h: float) -> Tuple[int, int]:
    x_px = int(round(nose_x + x_norm * face_h))
    y_px = int(round(nose_y + y_norm * face_h))
    return x_px, y_px


def draw_axes(img: np.ndarray, nose_x: float, nose_y: float, face_h: float, ui_scale: float = 1.0) -> None:
    h, w = img.shape[:2]
    cv2.line(img, (0, int(round(nose_y))), (w - 1, int(round(nose_y))), (180, 180, 180), 1)
    cv2.line(img, (int(round(nose_x)), 0), (int(round(nose_x)), h - 1), (180, 180, 180), 1)
    cv2.circle(img, (int(round(nose_x)), int(round(nose_y))), max(2, int(round(6 * ui_scale))), (0, 0, 255), -1)
    cv2.putText(
        img,
        f"nose=(0,0), face_h={face_h:.1f}px",
        (int(round(nose_x)) + int(round(12 * ui_scale)), int(round(nose_y)) - int(round(10 * ui_scale))),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6 * ui_scale,
        (30, 30, 30),
        scaled_thickness(1, ui_scale),
        cv2.LINE_AA,
    )


def draw_mouth_rule_guides(
    img: np.ndarray,
    tc: ToolsClustering,
    nose_x: float,
    nose_y: float,
    face_h: float,
    ui_scale: float = 1.0,
) -> None:
    """Draw guide lines for mouth rule (not a rectangle test)."""
    h, w = img.shape[:2]

    # Rule guide 1: top must be >= 0 => mouth bbox starts at or below nose line.
    y_zero = int(round(nose_y))
    cv2.line(img, (0, y_zero), (w - 1, y_zero), (20, 140, 20), 2)

    # Rule guide 2: bbox must straddle x=0 and width <= MAX_FACE_WIDTH.
    max_x = tc.MAX_FACE_WIDTH
    x_left, _ = norm_to_px(-max_x, 0.0, nose_x, nose_y, face_h)
    x_right, _ = norm_to_px(max_x, 0.0, nose_x, nose_y, face_h)
    cv2.line(img, (x_left, 0), (x_left, h - 1), (20, 140, 20), 1)
    cv2.line(img, (x_right, 0), (x_right, h - 1), (20, 140, 20), 1)

    msg = f"mouth_rule: top>=0, straddle x=0, width<=MAX_FACE_WIDTH({tc.MAX_FACE_WIDTH:.2f})"
    cv2.putText(
        img,
        msg,
        (int(round(20 * ui_scale)), max(int(round(24 * ui_scale)), y_zero + int(round(24 * ui_scale)))),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55 * ui_scale,
        (20, 120, 20),
        scaled_thickness(1, ui_scale),
        cv2.LINE_AA,
    )


def extract_index_knuckles_from_body_landmarks(body_landmarks_normalized):
    """Extract left/right index-finger points from normalized body landmarks (19/20)."""
    left = right = None
    if body_landmarks_normalized is None:
        return left, right

    try:
        landmarks = body_landmarks_normalized.landmark if hasattr(body_landmarks_normalized, "landmark") else body_landmarks_normalized
        if landmarks is None:
            return left, right

        if len(landmarks) > 19:
            lm = landmarks[19]
            if hasattr(lm, "x") and hasattr(lm, "y"):
                left = [float(lm.x), float(lm.y), 0.0]
            elif isinstance(lm, (list, tuple, np.ndarray)) and len(lm) >= 2:
                left = [float(lm[0]), float(lm[1]), 0.0]

        if len(landmarks) > 20:
            lm = landmarks[20]
            if hasattr(lm, "x") and hasattr(lm, "y"):
                right = [float(lm.x), float(lm.y), 0.0]
            elif isinstance(lm, (list, tuple, np.ndarray)) and len(lm) >= 2:
                right = [float(lm[0]), float(lm[1]), 0.0]
    except Exception:
        return None, None

    return left, right


def draw_hand_intersection_areas(
    img: np.ndarray,
    tc: ToolsClustering,
    body_landmarks_normalized,
    nose_x: float,
    nose_y: float,
    face_h: float,
    alpha: float,
    ui_scale: float = 1.0,
) -> Dict[str, Optional[Tuple[float, float]]]:
    """Draw left/right hand touch and nearby regions used for hand assignment gating."""
    left_knuckle, right_knuckle = extract_index_knuckles_from_body_landmarks(body_landmarks_normalized)
    result = {
        "left": None,
        "right": None,
    }

    touch_r_px = int(round(tc.TOUCH_THRESHOLD * face_h))
    nearby_r_px = int(round(tc.TOUCH_THRESHOLD * 2.0 * face_h))

    def draw_hand(name: str, knuckle, color_touch: Color, color_near: Color):
        if not knuckle:
            return None
        if len(knuckle) < 2:
            return None
        if abs(float(knuckle[1]) - 8.0) < 1e-6 and abs(float(knuckle[0])) < 1e-6:
            # Default off-screen sentinel used in assignment logic.
            return None

        x_px, y_px = norm_to_px(float(knuckle[0]), float(knuckle[1]), nose_x, nose_y, face_h)

        overlay = img.copy()
        cv2.circle(overlay, (x_px, y_px), max(1, nearby_r_px), color_near, -1)
        cv2.addWeighted(overlay, alpha * 0.65, img, 1.0 - alpha * 0.65, 0.0, img)

        overlay = img.copy()
        cv2.circle(overlay, (x_px, y_px), max(1, touch_r_px), color_touch, -1)
        cv2.addWeighted(overlay, alpha * 0.85, img, 1.0 - alpha * 0.85, 0.0, img)

        cv2.circle(img, (x_px, y_px), max(1, nearby_r_px), color_near, 2)
        cv2.circle(img, (x_px, y_px), max(1, touch_r_px), color_touch, 2)
        cv2.circle(img, (x_px, y_px), 4, (0, 0, 0), -1)

        label = (
            f"{name}_hand: touch<= {tc.TOUCH_THRESHOLD:.2f}, nearby<= {tc.TOUCH_THRESHOLD*2.0:.2f} "
            f"(norm units)"
        )
        cv2.putText(
            img,
            label,
            (x_px + int(round(8 * ui_scale)), max(int(round(18 * ui_scale)), y_px - int(round(10 * ui_scale)))),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5 * ui_scale,
            color_touch,
            scaled_thickness(1, ui_scale),
            cv2.LINE_AA,
        )
        return (float(knuckle[0]), float(knuckle[1]))

    result["left"] = draw_hand("left", left_knuckle, (40, 120, 255), (120, 190, 255))
    result["right"] = draw_hand("right", right_knuckle, (255, 120, 40), (255, 190, 120))
    return result


def draw_shoulder_band(
    img: np.ndarray,
    tc: ToolsClustering,
    body_landmarks_normalized,
    nose_x: float,
    nose_y: float,
    face_h: float,
    alpha: float,
    ui_scale: float = 1.0,
) -> bool:
    """Draw the dynamic shoulder band used by is_shoulder_object. Returns True if drawn."""
    left_shoulder, right_shoulder = tc.extract_shoulder_points(body_landmarks_normalized)
    if left_shoulder is None or right_shoulder is None:
        return False

    x1, y1 = float(left_shoulder[0]), float(left_shoulder[1])
    x2, y2 = float(right_shoulder[0]), float(right_shoulder[1])

    # Match ToolsClustering logic: shoulder band extends 1.0 normalized unit lower.
    band_drop = 1.0
    p1 = norm_to_px(x1, y1, nose_x, nose_y, face_h)
    p2 = norm_to_px(x2, y2, nose_x, nose_y, face_h)
    p3 = norm_to_px(x2, y2 + band_drop, nose_x, nose_y, face_h)
    p4 = norm_to_px(x1, y1 + band_drop, nose_x, nose_y, face_h)

    pts = np.array([p1, p2, p3, p4], dtype=np.int32)
    overlay = img.copy()
    cv2.fillPoly(overlay, [pts], (255, 40, 170))
    cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0.0, img)
    cv2.polylines(img, [pts], isClosed=True, color=(180, 10, 120), thickness=2)

    cv2.putText(
        img,
        "shoulder_band (dynamic): lm11->lm12, +1.0y",
        (
            min(p1[0], p4[0]) + int(round(4 * ui_scale)),
            max(int(round(20 * ui_scale)), min(p1[1], p2[1]) - int(round(10 * ui_scale))),
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55 * ui_scale,
        (180, 10, 120),
        scaled_thickness(1, ui_scale),
        cv2.LINE_AA,
    )
    return True


def draw_rect_zone(
    img: np.ndarray,
    name: str,
    rect: Dict[str, float],
    color: Color,
    alpha: float,
    nose_x: float,
    nose_y: float,
    face_h: float,
    ui_scale: float = 1.0,
) -> None:
    x1, y1 = norm_to_px(rect["x_min"], rect["y_top"], nose_x, nose_y, face_h)
    x2, y2 = norm_to_px(rect["x_max"], rect["y_bottom"], nose_x, nose_y, face_h)
    left = min(x1, x2)
    right = max(x1, x2)
    top = min(y1, y2)
    bottom = max(y1, y2)

    overlay = img.copy()
    cv2.rectangle(overlay, (left, top), (right, bottom), color, -1)
    cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0.0, img)
    cv2.rectangle(img, (left, top), (right, bottom), color, 2)

    label = (
        f"{name} x[{rect['x_min']:.2f},{rect['x_max']:.2f}] "
        f"y[{rect['y_top']:.2f},{rect['y_bottom']:.2f}]"
    )
    label_y = max(int(round(22 * ui_scale)), top - int(round(8 * ui_scale)))
    cv2.putText(
        img,
        label,
        (left + int(round(4 * ui_scale)), label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5 * ui_scale,
        color,
        scaled_thickness(1, ui_scale),
        cv2.LINE_AA,
    )


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def build_output_path(output_path: str, image_id: Optional[int]) -> str:
    if image_id is None:
        return output_path
    base, ext = os.path.splitext(output_path)
    if not ext:
        ext = ".png"
    return f"{base}_{image_id}{ext}"


def build_batch_output_path(output_path: str, image_id: int) -> str:
    output_dir = os.path.dirname(output_path)
    output_name = os.path.basename(output_path)
    batch_dir = os.path.join(output_dir, BATCH_OUTPUT_SUBDIR) if output_dir else BATCH_OUTPUT_SUBDIR
    return build_output_path(os.path.join(batch_dir, output_name), image_id)


def unique_image_ids(image_ids):
    seen = set()
    for image_id in image_ids:
        image_id = int(image_id)
        if image_id in seen:
            continue
        seen.add(image_id)
        yield image_id


def render_one(args, tc: ToolsClustering, zones: Dict[str, Dict[str, float]], image_id: Optional[int], output_path: str) -> None:
    tc = ToolsClustering(CLUSTER_TYPE="ObjectFusion", VERBOSE=False, session=None)
    db_ctx = None
    if image_id is not None:
        db_ctx = fetch_image_context_from_mysql(image_id)

    image_path = args.image
    if image_path is None and db_ctx is not None:
        image_path = db_ctx.get("resolved_local_path")

    ui_scale = 1.0
    if image_path:
        img = cv2.imread(image_path)
        if img is None:
            raise RuntimeError(f"Failed to read image: {image_path}")
        if UP_RES_4X:
            original_h, original_w = img.shape[:2]
            img = upscale_image_4x(img, UPSCALE_MODEL_PATH)
            upscaled_h, upscaled_w = img.shape[:2]
            if original_w > 0 and original_h > 0 and (upscaled_w != original_w or upscaled_h != original_h):
                ui_scale = upscaled_w / float(original_w)
    else:
        img = np.full((args.height, args.width, 3), 255, dtype=np.uint8)

    h, w = img.shape[:2]

    db_nose_x = float(db_ctx["nose_pixel_x"]) if db_ctx and db_ctx.get("nose_pixel_x") is not None else None
    db_nose_y = float(db_ctx["nose_pixel_y"]) if db_ctx and db_ctx.get("nose_pixel_y") is not None else None
    db_face_h = float(db_ctx["face_height"]) if db_ctx and db_ctx.get("face_height") is not None else None

    nose_x = float(args.nose_x) if args.nose_x is not None else (db_nose_x if db_nose_x is not None else (w / 2.0))
    nose_y = float(args.nose_y) if args.nose_y is not None else (db_nose_y if db_nose_y is not None else (h * 0.18))
    face_h = float(args.face_height) if args.face_height is not None else (db_face_h if db_face_h is not None else (min(w, h) / 6.0))

    # Keep overlay geometry proportional when image super-resolution is applied.
    if ui_scale != 1.0:
        nose_x *= ui_scale
        nose_y *= ui_scale
        face_h *= ui_scale

    draw_axes(img, nose_x, nose_y, face_h, ui_scale=ui_scale)
    draw_mouth_rule_guides(img, tc, nose_x, nose_y, face_h, ui_scale=ui_scale)

    palette: Dict[str, Color] = {
        "left_eye_zone": (255, 120, 60),
        "right_eye_zone": (60, 180, 255),
        "under_eye_zone": (255, 200, 40),
        "eye_cover_zone": (180, 80, 255),
        "forehead_zone": (80, 220, 120),
        "waist_zone": (70, 70, 255),
        "feet_zone": (40, 180, 80),
    }

    for zone_name, rect in zones.items():
        draw_rect_zone(
            img,
            zone_name,
            rect,
            palette.get(zone_name, (120, 120, 120)),
            args.alpha,
            nose_x,
            nose_y,
            face_h,
            ui_scale,
        )

    shoulder_drawn = False
    if db_ctx is not None and db_ctx.get("body_landmarks_normalized") is not None:
        shoulder_drawn = draw_shoulder_band(
            img,
            tc,
            db_ctx.get("body_landmarks_normalized"),
            nose_x,
            nose_y,
            face_h,
            args.alpha,
            ui_scale,
        )

    hand_regions = {"left": None, "right": None}
    if db_ctx is not None and db_ctx.get("body_landmarks_normalized") is not None:
        hand_regions = draw_hand_intersection_areas(
            img,
            tc,
            db_ctx.get("body_landmarks_normalized"),
            nose_x,
            nose_y,
            face_h,
            args.alpha,
            ui_scale,
        )

    ensure_parent_dir(output_path)
    cv2.imwrite(output_path, img)

    print("Done.")
    print(f"Wrote: {output_path}")
    print(f"nose_x={nose_x:.1f}, nose_y={nose_y:.1f}, face_height={face_h:.1f}")
    if db_ctx is not None:
        print("DB context:")
        print(f"  image_id={db_ctx.get('image_id')}")
        print(f"  site_name_id={db_ctx.get('site_name_id')}")
        print(f"  imagename={db_ctx.get('imagename')}")
        print(f"  contentUrl={db_ctx.get('contentUrl')}")
        print(f"  resolved_local_path={db_ctx.get('resolved_local_path')}")
        print(f"  encoding_id={db_ctx.get('encoding_id')}")
        print(f"  is_face={db_ctx.get('is_face')}")
        print(f"  shoulder_band_drawn={shoulder_drawn}")
        print(f"  left_hand_region_knuckle={hand_regions.get('left')}")
        print(f"  right_hand_region_knuckle={hand_regions.get('right')}")
    print("Zones (from ToolsClustering):")
    for name, rect in zones.items():
        print(
            f"  {name}: x[{rect['x_min']:.2f},{rect['x_max']:.2f}] "
            f"y[{rect['y_top']:.2f},{rect['y_bottom']:.2f}]"
        )


def main():
    args = parse_args()

    tc = ToolsClustering(CLUSTER_TYPE="ObjectFusion", VERBOSE=False, session=None)
    zones = get_zone_rects(tc)

    if INPUT_LIST:
        for image_id in unique_image_ids(INPUT_LIST):
            render_one(args, tc, zones, image_id, build_batch_output_path(args.output, image_id))
        return

    render_one(args, tc, zones, args.image_id, build_output_path(args.output, args.image_id))


if __name__ == "__main__":
    main()
