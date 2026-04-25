from sqlalchemy import create_engine, select, text, bindparam, Column, Integer, Float, ForeignKey, BLOB
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool
from my_declarative_base import Base, Images, Detections, Encodings
import pickle
import numpy as np
import json
import os
import pandas as pd
import hashlib

class ToolsClustering:
    """Store key clustering info for use across codebase"""

    def __init__(self, CLUSTER_TYPE, VERBOSE=False, session=None):
        self.VERBOSE = VERBOSE
        self.CLUSTER_TYPE = CLUSTER_TYPE
        self.CLUSTER_MEDIANS = None
        self.session = session
        # Object-hand relationship constants
        self.TOUCH_THRESHOLD = 0.5  # face height units
        self.CLASS_ID_WEIGHT = 10  # OLD WAY multiplier to give more weight to class_id in clustering (since it's categorical and we want it to separate well)
        self.OVERLAP_IOU_THRESHOLD = 0.5
        self.HIGH_CONFIDENCE_THRESHOLD = 0.9
        self.CONFIDENCE_DIFF_THRESHOLD = 0.3
        self.MIN_DETECTION_CONFIDENCE = 0.4
        self.DEFAULT_HAND_POSITION = [0.0, 8.0, 0.0]
        self.TIE_CLASS_ID = 27
        self.USE_ALLOWLIST = True
        self.FLOWER_CLASSES = {104, 105, 106, 107}
        self.HAND_ONLY_CLASSES = {108, 109}
        self.COVID_MASK_CLASSES = {110}
        self.FULL_FACE_TOP_BIASED_CLASSES = {111, 112}
        self.UNDER_EYE_CLASSES = {113}
        self.EYE_OR_FOREHEAD_CLASSES = {114}
        self.EYE_ONLY_CLASSES = {115}
        self.HAND_OR_EYE_CLASSES = {116, 117, 118, 119}
        self.HANDHELD_LIKE_CLASSES = {39, 40, 41, 67, 73, 76, 77, 79, 80, 82, 95}
        # Lower-body guardrail for handheld-like classes.
        # Goal: reduce implausible feet/waist pulls while preserving clear lower-body cases.
        self.SMALL_HANDHELD_CLASSES = {67, 82, 95}  # legacy stricter subset
        self.SMALL_HANDHELD_LOWER_BODY_CONF_PENALTY = 0.24
        self.SMALL_HANDHELD_LOWER_BODY_DISTANCE_MARGIN = 0.35
        self.HANDHELD_LIKE_LOWER_BODY_MIN_SCORE = 0.62
        self.LOWER_BODY_VISIBILITY_MIN = 0.45
        self.LOWER_BODY_OCCLUSION_PENALTY = 0.22
        self.CLASS67_EXTRA_LOWER_BODY_PENALTY = {
            'waist': 0.10,
            'feet': 0.20,
        }
        self.COMPATIBILITY_SLOT_COLUMNS = (
            'hand', 'left_eye', 'right_eye', 'top_face', 'mouth', 'shoulder', 'waist', 'feet'
        )
        self.COMPATIBILITY_SCORE_BIAS = {
            0: -9999.0,  # hard reject
            1: -0.12,    # de-emphasize
            2: 0.06,     # prefer
        }
        self.COMPATIBILITY_MATRIX_PATH = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'utilities','data',
            'object_slot_compatibility_matrix.csv',
        )
        self.DEBUG_TRACKED_CLASS_IDS = tuple(range(110, 120))
        self._class_assignment_slot_names = (
            'left_hand_object',
            'right_hand_object',
            'top_face_object',
            'left_eye_object',
            'right_eye_object',
            'mouth_object',
            'shoulder_object',
            'waist_object',
            'feet_object',
        )
        all_class_ids = set(range(0, 120))
        self._allowlist_slots = tuple(self.COMPATIBILITY_SLOT_COLUMNS)
        self._allowlist_reject_counts = {slot: 0 for slot in self._allowlist_slots}
        self.compatibility_matrix = self._load_compatibility_matrix_from_csv(all_class_ids)
        self._slot_unassigned_reason_counts = {
            'waist': {},
            'feet': {},
        }
        self.reset_class_assignment_debug_counts()
        self.reset_class_pipeline_debug_counts()
        # Face object constraints to avoid large background objects
        self.MAX_FACE_WIDTH = 2.0  # max width of left+right to be considered face object
        self.MAX_FACE_VERT_EXTENSION = 0.75  # max how far object can extend into opposite zone
        self.EYE_ZONE_TOP = -0.65
        self.EYE_ZONE_BOTTOM = 0.15
        self.LEFT_EYE_X_MIN = -0.9
        self.LEFT_EYE_X_MAX = -0.05
        self.RIGHT_EYE_X_MIN = 0.05
        self.RIGHT_EYE_X_MAX = 0.9
        # Tie-specific neck/chest fallback geometry used when shoulder landmarks are missing.
        self.TIE_NECK_MAX_WIDTH = 1.25
        self.TIE_NECK_TOP_MAX = 1.25
        self.TIE_NECK_BOTTOM_MIN = 0.35
        self.TIE_NECK_BOTTOM_MAX = 3.40
        self.TIE_NECK_MIN_HEIGHT = 0.45
        self.TIE_NECK_MIN_ASPECT_RATIO = 1.10
        self.TIE_NECK_CENTER_X_TOL = 0.35
        self.CENTERLINE_X = 0.0
        self.CENTERLINE_Y = 0.0
        self.MOUTH_MAX_TOP_EXTENSION = 0.0
        self.FULL_FACE_MASK_X_MIN = -0.15
        self.FULL_FACE_MASK_X_MAX = 0.15
        self.SHOULDER_BAND_EXTENSION = 1.0
        self.COVID_MASK_TOP_MIN = -0.4
        self.COVID_MASK_TOP_MAX = 0.5
        self.COVID_MASK_BOTTOM_MIN = 0.3
        self.COVID_MASK_BOTTOM_MAX = .8
        # Mask intent scoring constants (class 110) -- Rule Spec v1.0
        self.MASK_INTENT_WORN_MIN = 0.60
        self.MASK_INTENT_HELD_MIN = 0.60
        self.MASK_INTENT_MARGIN = 0.12
        self.MASK_ALLOW_DUAL_MOUTH_HAND = True
        self.MASK_MOUTH_STICKY_MARGIN = 0.08
        self.W_MASK_MOUTH_GEOM = 0.40
        self.W_MASK_FACE_ANCHOR = 0.25
        self.W_MASK_CENTERLINE = 0.15
        self.W_MASK_HAND_PROX = 0.20
        self.W_MASK_DUAL_HAND = 0.10
        self.W_MASK_OFF_FACE = 0.25
        # Classes 108-109 lower-body eligibility constants -- Rule Spec v1.0
        self.CLASS108109_HAND_PREFERENCE_BONUS = 0.12
        self.CLASS108109_ENABLE_SHOULDER = True
        self.CLASS108109_ENABLE_WAIST = True
        self.CLASS108109_ENABLE_FEET = True
        self.CLASS108109_SHOULDER_MIN_INTERSECT = 0.12
        self.CLASS108109_WAIST_MIN_INTERSECT = 0.18
        self.CLASS108109_FEET_MIN_INTERSECT = 0.22
        self.CLASS108109_WAIST_MIN_SCORE = 0.58
        self.CLASS108109_FEET_MIN_SCORE = 0.60
        self.CLASS108109_NEAR_HAND_DIST = 0.45
        self.CLASS108109_LOWER_BODY_NEAR_HAND_PENALTY = 0.18
        self.FULL_FACE_MASK_TOP_MAX = -0.15
        self.FULL_FACE_MASK_BOTTOM_MIN = 0.15
        self.UNDER_EYE_ZONE_TOP = -0.45
        self.UNDER_EYE_ZONE_BOTTOM = 0.45
        self.EYE_COVER_ZONE_TOP = -0.80
        self.EYE_COVER_ZONE_BOTTOM = 0.30
        self.FOREHEAD_X_MIN = -1.00
        self.FOREHEAD_X_MAX = 1.00
        self.FOREHEAD_ZONE_TOP = -1.10
        self.FOREHEAD_ZONE_BOTTOM = -0.10
        self.WAIST_ZONE_TOP = 2.5
        self.WAIST_ZONE_BOTTOM = 4.5
        self.WAIST_X_MIN = -1.40
        self.WAIST_X_MAX = 1.40
        self.FEET_ZONE_TOP = 4.75
        self.FEET_ZONE_BOTTOM = 10.20
        self.FEET_X_MIN = -2.00
        self.FEET_X_MAX = 2.00
        
        # Feature standardization settings for ObjectFusion
        self.USE_FEATURE_STANDARDIZATION = True  # Use StandardScaler to normalize all features to similar scale
        self.SUPPRESS_ARMS_FEATURES = False

        # baseline
        # self.FEATURE_WEIGHTS = {
        #     'face_angle': .5,      # pitch, yaw, roll - LOW weight (prevent face-angle mega-clusters)
        #     'class_id': 3.0,        # class_id - VERY HIGH weight at RAW SCALE (0-107) to force object-type separation
        #     'confidence': 0.5,      # detection confidence - very low weight
        #     'bbox': 2.0,            # bbox coordinates - reduced to standard weight
        #     'has_object': 1.0,      # binary indicator - high weight but lower than class_id
        # }

        # # Class-separation stronger
        # self.FEATURE_WEIGHTS = {
        #     'face_angle': .5,      # pitch, yaw, roll - LOW weight (prevent face-angle mega-clusters)
        #     'class_id': 5.0,        # class_id - VERY HIGH weight at RAW SCALE (0-107) to force object-type separation
        #     'confidence': 0.5,      # detection confidence - very low weight
        #     'bbox': 2.0,            # bbox coordinates - reduced to standard weight
        #     'has_object': 1.2,      # binary indicator - high weight but lower than class_id
        # }

        # # Spatial tighter
        # self.FEATURE_WEIGHTS = {
        #     'face_angle': .5,      # pitch, yaw, roll - LOW weight (prevent face-angle mega-clusters)
        #     'class_id': 3.0,        # class_id - VERY HIGH weight at RAW SCALE (0-107) to force object-type separation
        #     'confidence': 0.5,      # detection confidence - very low weight
        #     'bbox': 2.6,            # bbox coordinates - reduced to standard weight
        #     'has_object': 1.0,      # binary indicator - high weight but lower than class_id
        # }

        # Spatial looser
        # self.FEATURE_WEIGHTS = {
        #     'face_angle': .5,      # pitch, yaw, roll - LOW weight (prevent face-angle mega-clusters)
        #     'class_id': 3.0,        # class_id - VERY HIGH weight at RAW SCALE (0-107) to force object-type separation
        #     'confidence': 0.5,      # detection confidence - very low weight
        #     'bbox': 1.4,            # bbox coordinates - reduced to standard weight
        #     'has_object': 1.0,      # binary indicator - high weight but lower than class_id
        # }

        # R1_object_up_2x
        self.FEATURE_WEIGHTS = {
            'face_angle': .5,      # pitch, yaw, roll - LOW weight (prevent face-angle mega-clusters)
            'class_id': 9.0,        # class_id - VERY HIGH weight at RAW SCALE (0-107) to force object-type separation
            'confidence': 1.0,      # detection confidence - very low weight
            'bbox': 4.0,            # bbox coordinates - reduced to standard weight
            'has_object': 2.0,      # binary indicator - high weight but lower than class_id
        }

        # Store fitted scaler for inverse transform during median calculation
        self.feature_scaler = None
        self.ARMS_POSE_CACHE_TABLE = 'ImagesArmsFeatures3D'
        self.ARMS_POSE_SUBSET_NAME = 'arms_0_22_xyz'
        self.ARMS_POSE_SUBSET_VERSION = 1
        self.WORLD_LMS_REPORT_MAX_SAMPLES = 20
        self.world_lms_stats = self._new_world_lms_stats()
        self.CLUSTER_DATA = {
            "BodyPoses": {"data_column": "mongo_body_landmarks", "is_feet": 1, "mongo_hand_landmarks": None},
            "BodyPoses3D": {"data_column": "mongo_body_landmarks_3D", "is_feet": 1, "mongo_hand_landmarks": None}, # changed this for testing
            "ArmsPoses3D": {"data_column": "mongo_body_landmarks_3D", "is_feet": None, "mongo_hand_landmarks": 1},
            "ObjectFusion": {"data_column": "mongo_body_landmarks_norm", "is_feet": None, "mongo_hand_landmarks": None},
            "HandsGestures": {"data_column": "mongo_hand_landmarks", "is_feet": None, "mongo_hand_landmarks": 1},
            "HandsPositions": {"data_column": "mongo_hand_landmarks_norm", "is_feet": None, "mongo_hand_landmarks": 1},
            "FingertipsPositions": {"data_column": "mongo_hand_landmarks_norm", "is_feet": None, "mongo_hand_landmarks": 1},
            "HSV": {"data_column": ["hue", "sat", "val"], "is_feet": None},
}
        self.OBJECT_SIGNATURE_SLOT_MAP = [
            ("left_hand_object", "LH"),
            ("right_hand_object", "RH"),
            ("top_face_object", "TF"),
            ("left_eye_object", "LE"),
            ("right_eye_object", "RE"),
            ("mouth_object", "MO"),
            ("shoulder_object", "SH"),
            ("waist_object", "WA"),
            ("feet_object", "FT"),
        ]
        self.SIGNATURE_HEAD_MIN_SUPPORT = 200
        self.SIGNATURE_FALLBACK_MAX_SLOTS = 4
        self.SIGNATURE_FALLBACK_PRIORITY = {
            "left_hand_object": 80.0,
            "right_hand_object": 80.0,
            "top_face_object": 35.0,
            "left_eye_object": 30.0,
            "right_eye_object": 30.0,
            "mouth_object": 30.0,
            "shoulder_object": 25.0,
            "waist_object": 20.0,
            "feet_object": -9999.0,
        }
        self.SIGNATURE_COMPATIBILITY_FALLBACK_BONUS = {
            0: -9999.0,
            1: 1.0,
            2: 5.0,
        }
        self.SIGNATURE_FALLBACK_CONF_WEIGHT = 10.0

    # ==================== OBJECT SIGNATURE METHODS ====================

    def extract_slot_class_id(self, slot_value):
        """Extract class_id int from one slot payload; return 0 for empty/invalid."""
        if slot_value is None:
            return 0

        try:
            if isinstance(slot_value, dict):
                class_id = slot_value.get('class_id')
                if class_id is None:
                    return 0
                return int(float(class_id))

            if isinstance(slot_value, (list, tuple, np.ndarray)):
                if len(slot_value) == 0:
                    return 0
                first = slot_value[0]
                if first is None:
                    return 0
                return int(float(first))

            if pd.isna(slot_value):
                return 0
            return int(float(slot_value))
        except Exception:
            return 0

    def build_slot_signature_fields(self, row):
        """Build deterministic token/hash/object-count for a row."""
        token_parts = []
        n_objects = 0

        for slot_col, slot_label in self.OBJECT_SIGNATURE_SLOT_MAP:
            class_id = self.extract_slot_class_id(row.get(slot_col))
            if class_id > 0:
                n_objects += 1
            token_parts.append(f"{slot_label}:{class_id}")

        token = "|".join(token_parts)
        sig_hash = hashlib.sha1(token.encode('utf-8')).hexdigest()
        return token, sig_hash, n_objects

    def append_object_signature_fields(self, df):
        """Append slot_signature_token/hash/n_objects columns to DataFrame."""
        required_slots = [slot for slot, _ in self.OBJECT_SIGNATURE_SLOT_MAP]
        missing_slots = [col for col in required_slots if col not in df.columns]
        if missing_slots:
            print(f"[SIGNATURE] Missing slot columns, skipping signature build: {missing_slots}")
            return df

        if 'image_id' not in df.columns:
            print("[SIGNATURE] Missing image_id, skipping signature build.")
            return df

        out_df = df.copy()
        sig_values = out_df.apply(self.build_slot_signature_fields, axis=1, result_type='expand')
        sig_values.columns = ['slot_signature_token', 'slot_signature_hash', 'slot_signature_n_objects']
        out_df[['slot_signature_token', 'slot_signature_hash', 'slot_signature_n_objects']] = sig_values
        return out_df

    def extract_slot_confidence(self, slot_value):
        """Extract confidence float from one slot payload; return 0.0 for empty/invalid."""
        if slot_value is None:
            return 0.0

        try:
            if isinstance(slot_value, dict):
                conf_val = slot_value.get('conf', 0.0)
                if conf_val is None:
                    return 0.0
                return float(conf_val)

            if isinstance(slot_value, (list, tuple, np.ndarray)):
                if len(slot_value) > 1 and slot_value[1] is not None:
                    return float(slot_value[1])
                return 0.0

            return 0.0
        except Exception:
            return 0.0

    def _slot_name_for_compatibility(self, slot_col):
        if slot_col in ('left_hand_object', 'right_hand_object'):
            return 'hand'
        return slot_col.replace('_object', '')

    def build_fallback_signature_fields(self, row, max_slots=None):
        """
        Build fallback token/hash/n_objects for low-support signatures.
        Rules:
        - Exclude feet first.
        - Keep shoulder only when class_id == 27 (tie/neck case); otherwise drop shoulder.
        - Prioritize tie/neck class 27 in shoulder slot, then hands, then everything else.
        - Use compatibility matrix tier and confidence to break ties.
        """
        candidates = []
        selected_slots = set()
        slot_limit = self.SIGNATURE_FALLBACK_MAX_SLOTS if max_slots is None else max(int(max_slots), 0)

        for slot_col, _slot_label in self.OBJECT_SIGNATURE_SLOT_MAP:
            slot_value = row.get(slot_col)
            class_id = self.extract_slot_class_id(slot_value)
            if class_id <= 0:
                continue

            if slot_col == 'feet_object':
                continue

            # Neck/shoulder policy: keep only class 27; demote all other shoulder placements.
            if slot_col == 'shoulder_object' and class_id != self.TIE_CLASS_ID:
                continue

            slot_name = self._slot_name_for_compatibility(slot_col)
            compat_level = self._get_compatibility_level(class_id, slot_name)
            compat_bonus = float(self.SIGNATURE_COMPATIBILITY_FALLBACK_BONUS.get(compat_level, 0.0))
            if compat_bonus <= -9999.0:
                continue

            base_priority = float(self.SIGNATURE_FALLBACK_PRIORITY.get(slot_col, 10.0))
            if slot_col == 'shoulder_object' and class_id == self.TIE_CLASS_ID:
                # User-priority: neck+27 is highest-priority retained placement.
                base_priority = 100.0

            conf = self.extract_slot_confidence(slot_value)
            score = base_priority + compat_bonus + (max(conf, 0.0) * self.SIGNATURE_FALLBACK_CONF_WEIGHT)
            candidates.append((score, slot_col, class_id))

        # Prefer highest-priority compatible placements, capped for stronger tail collapse.
        candidates.sort(key=lambda x: (-x[0], x[1], x[2]))
        for _score, slot_col, _class_id in candidates[:slot_limit]:
            selected_slots.add(slot_col)

        token_parts = []
        n_objects = 0
        for slot_col, slot_label in self.OBJECT_SIGNATURE_SLOT_MAP:
            class_id = 0
            if slot_col in selected_slots:
                class_id = self.extract_slot_class_id(row.get(slot_col))
                if class_id > 0:
                    n_objects += 1
            token_parts.append(f"{slot_label}:{class_id}")

        token = "|".join(token_parts)
        sig_hash = hashlib.sha1(token.encode('utf-8')).hexdigest()
        return token, sig_hash, n_objects

    def persist_object_signatures(self, df, engine, batch_size=5000):
        """
        Persist dictionary (ObjectSignatures) and image assignments (ImagesObjectSignatures).
        Requires columns: image_id, slot_signature_token, slot_signature_hash, slot_signature_n_objects.
        """
        required_cols = ['image_id', 'slot_signature_token', 'slot_signature_hash', 'slot_signature_n_objects']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"[SIGNATURE] Missing required columns, skipping DB save: {missing_cols}")
            return {
                'unique_signatures': 0,
                'assignments_total': 0,
                'assignments_written': 0,
                'missing_hashes': 0,
                'skipped': True,
            }

        fallback_required_slots = [slot for slot, _ in self.OBJECT_SIGNATURE_SLOT_MAP]
        fallback_slot_cols = [col for col in fallback_required_slots if col in df.columns]
        persist_columns = required_cols + [col for col in fallback_slot_cols if col not in required_cols]

        persist_df = df[persist_columns].copy()
        persist_df = persist_df.dropna(subset=['image_id', 'slot_signature_hash', 'slot_signature_token'])
        if len(persist_df) == 0:
            print("[SIGNATURE] No rows to persist after dropna.")
            return {
                'unique_signatures': 0,
                'assignments_total': 0,
                'assignments_written': 0,
                'missing_hashes': 0,
                'skipped': True,
            }

        persist_df['image_id'] = persist_df['image_id'].astype(int)
        persist_df['slot_signature_n_objects'] = persist_df['slot_signature_n_objects'].fillna(0).astype(int)

        # Head/tail routing for signatures:
        # - Head signatures (>= SIGNATURE_HEAD_MIN_SUPPORT) keep original token/hash.
        # - Tail signatures (< SIGNATURE_HEAD_MIN_SUPPORT) are iteratively remapped via fallback token rules
        #   until they either reach the support floor or collapse to the all-Nones signature.
        signature_counts = persist_df['slot_signature_hash'].value_counts(dropna=False)
        persist_df['slot_signature_support'] = persist_df['slot_signature_hash'].map(signature_counts).fillna(0).astype(int)
        tail_mask = persist_df['slot_signature_support'] < int(self.SIGNATURE_HEAD_MIN_SUPPORT)

        fallback_applied_rows = 0
        if fallback_slot_cols and tail_mask.any():
            persist_df['_fallback_slot_budget'] = -1
            persist_df.loc[tail_mask, '_fallback_slot_budget'] = int(self.SIGNATURE_FALLBACK_MAX_SLOTS)

            while True:
                active_fallback_mask = tail_mask & (persist_df['_fallback_slot_budget'] >= 0)
                if not active_fallback_mask.any():
                    break

                fallback_values = persist_df.loc[active_fallback_mask].apply(
                    lambda row: self.build_fallback_signature_fields(
                        row,
                        max_slots=int(row['_fallback_slot_budget']),
                    ),
                    axis=1,
                    result_type='expand',
                )
                fallback_values.columns = ['slot_signature_token', 'slot_signature_hash', 'slot_signature_n_objects']
                persist_df.loc[active_fallback_mask, ['slot_signature_token', 'slot_signature_hash', 'slot_signature_n_objects']] = fallback_values

                signature_counts = persist_df['slot_signature_hash'].value_counts(dropna=False)
                persist_df['slot_signature_support'] = persist_df['slot_signature_hash'].map(signature_counts).fillna(0).astype(int)

                unresolved_mask = (
                    tail_mask
                    & (persist_df['slot_signature_support'] < int(self.SIGNATURE_HEAD_MIN_SUPPORT))
                    & (persist_df['_fallback_slot_budget'] > 0)
                )
                if not unresolved_mask.any():
                    break

                persist_df.loc[unresolved_mask, '_fallback_slot_budget'] = (
                    persist_df.loc[unresolved_mask, '_fallback_slot_budget'].astype(int) - 1
                )

            fallback_applied_rows = int(tail_mask.sum())
            persist_df = persist_df.drop(columns=['_fallback_slot_budget'])

        persist_df['slot_signature_n_objects'] = persist_df['slot_signature_n_objects'].fillna(0).astype(int)

        dict_rows = (
            persist_df[['slot_signature_hash', 'slot_signature_token', 'slot_signature_n_objects']]
            .drop_duplicates(subset=['slot_signature_hash'])
            .to_dict(orient='records')
        )
        assignments = persist_df[['image_id', 'slot_signature_hash']].to_dict(orient='records')

        insert_signature_sql = text(
            """
            INSERT INTO ObjectSignatures (
                slot_signature_hash,
                slot_signature_token,
                slot_signature_n_objects
            ) VALUES (
                :slot_signature_hash,
                :slot_signature_token,
                :slot_signature_n_objects
            )
            ON DUPLICATE KEY UPDATE
                slot_signature_token = VALUES(slot_signature_token),
                slot_signature_n_objects = VALUES(slot_signature_n_objects)
            """
        )

        insert_image_signature_sql = text(
            """
            INSERT INTO ImagesObjectSignatures (
                image_id,
                cluster_id
            ) VALUES (
                :image_id,
                :cluster_id
            )
            ON DUPLICATE KEY UPDATE
                cluster_id = VALUES(cluster_id)
            """
        )

        print(
            f"[SIGNATURE] Upserting {len(dict_rows)} unique signatures and {len(assignments)} image assignments... "
            f"fallback_applied_rows={fallback_applied_rows} head_min={self.SIGNATURE_HEAD_MIN_SUPPORT}"
        )

        with engine.begin() as conn:
            for start in range(0, len(dict_rows), batch_size):
                batch = dict_rows[start:start + batch_size]
                conn.execute(insert_signature_sql, batch)
                # Seed cluster_id=0 as the canonical "all-Nones" signature before any auto-increment rows
                none_row = {slot: None for slot, _ in self.OBJECT_SIGNATURE_SLOT_MAP}
                none_token, none_hash, _ = self.build_slot_signature_fields(none_row)
                seed_sql = text(
                    """
                    INSERT IGNORE INTO ObjectSignatures (cluster_id, slot_signature_hash, slot_signature_token, slot_signature_n_objects)
                    VALUES (0, :slot_signature_hash, :slot_signature_token, 0)
                    """
                )
                conn.execute(seed_sql, {'slot_signature_hash': none_hash, 'slot_signature_token': none_token})

            unique_hashes = sorted({row['slot_signature_hash'] for row in assignments})
            hash_to_cluster_id = {}
            select_map_sql = text(
                """
                SELECT cluster_id, slot_signature_hash
                FROM ObjectSignatures
                WHERE slot_signature_hash IN :hash_list
                """
            ).bindparams(bindparam('hash_list', expanding=True))

            for start in range(0, len(unique_hashes), batch_size):
                hash_batch = unique_hashes[start:start + batch_size]
                rows = conn.execute(select_map_sql, {'hash_list': hash_batch}).fetchall()
                for row in rows:
                    hash_to_cluster_id[row.slot_signature_hash] = int(row.cluster_id)

            image_rows = []
            missing_hashes = 0
            for assignment in assignments:
                cluster_id = hash_to_cluster_id.get(assignment['slot_signature_hash'])
                if cluster_id is None:
                    missing_hashes += 1
                    continue
                image_rows.append({'image_id': int(assignment['image_id']), 'cluster_id': cluster_id})

            if missing_hashes > 0:
                print(f"[SIGNATURE] Warning: missing cluster_id for {missing_hashes} hash rows.")

            for start in range(0, len(image_rows), batch_size):
                batch = image_rows[start:start + batch_size]
                conn.execute(insert_image_signature_sql, batch)

        print(f"[SIGNATURE] Persist complete. image_rows_written={len(image_rows)}")
        return {
            'unique_signatures': len(dict_rows),
            'assignments_total': len(assignments),
            'assignments_written': len(image_rows),
            'missing_hashes': missing_hashes,
            'fallback_applied_rows': fallback_applied_rows,
            'head_min_support': int(self.SIGNATURE_HEAD_MIN_SUPPORT),
            'skipped': False,
        }

    def print_signature_run_summary(self, rows_in, signature_stats, signatures_only=True):
        """Print standardized summary for signature population runs."""
        print("\n=== SIGNATURE RUN SUMMARY ===")
        print(f"rows_in: {rows_in}")
        print(f"unique_signatures: {signature_stats.get('unique_signatures', 0)}")
        print(f"assignments_total: {signature_stats.get('assignments_total', 0)}")
        print(f"assignments_written: {signature_stats.get('assignments_written', 0)}")
        print(f"missing_hashes: {signature_stats.get('missing_hashes', 0)}")
        print(f"fallback_applied_rows: {signature_stats.get('fallback_applied_rows', 0)}")
        print(f"head_min_support: {signature_stats.get('head_min_support', self.SIGNATURE_HEAD_MIN_SUPPORT)}")
        if signatures_only:
            print("Skipping KMeans/ObjectFusion cluster writes because SIGNATURES_ONLY=True")
        print("=== END SIGNATURE RUN SUMMARY ===\n")

    def _new_world_lms_stats(self):
        """Create a fresh world-landmark stats payload for ObjectFusion runs."""
        return {
            'candidates_total': 0,
            'world_lms_present': 0,
            'world_lms_missing': 0,
            'world_lms_missing_sample_ids': [],
            'excluded_missing_world_lms': 0,
            'excluded_invalid_subset': 0,
            'excluded_invalid_subset_sample_ids': [],
            'rows_no_object_retained': 0,
        }

    def reset_world_lms_stats(self):
        """Reset world-landmark stats at run start."""
        self.world_lms_stats = self._new_world_lms_stats()
        return self.world_lms_stats

    def append_world_lms_sample_id(self, sample_key, image_id):
        """Append one sample image_id to a stats list with max-size guard."""
        sample_list = self.world_lms_stats.get(sample_key)
        if not isinstance(sample_list, list):
            return
        if image_id is None:
            return
        if len(sample_list) >= self.WORLD_LMS_REPORT_MAX_SAMPLES:
            return
        try:
            sample_list.append(int(image_id))
        except Exception:
            return

    def increment_world_lms_stat(self, stat_key, amount=1):
        """Increment a numeric world-landmark stat in-place."""
        if stat_key not in self.world_lms_stats:
            return
        try:
            self.world_lms_stats[stat_key] += int(amount)
        except Exception:
            return

    def get_cluster_medians(self, session, Clusters, USE_SUBSET_MEDIANS=False, SUBSET_LANDMARKS=None):
        print("getting cluster medians")
        # Create a SQLAlchemy select statement
        select_query = select(Clusters.cluster_id, Clusters.cluster_median)

        # Execute the query using your SQLAlchemy session
        results = session.execute(select_query)
        median_dict = {}

        # Process the results as needed
        # print(f'Found {len(results)} clusters with medians.')
        # print("results is a sqlalchemy.engine.result.ChunkedIteratorResult object at 0x3233d5400. this is how many: ",len(results))
        for row in results:
            # print(row)
            cluster_id, cluster_median_pickle = row
            # print("cluster_id: ",cluster_id)

            # import pprint
            # pp = pprint.PrettyPrinter(indent=4)
            # print the pickle using pprint
            # pp.pprint(cluster_median_pickle)

            cluster_median = pickle.loads(cluster_median_pickle)
            # print(f"cluster_median {cluster_id}: ", cluster_median)
            # Check the type and content of the deserialized object
            if not isinstance(cluster_median, np.ndarray):
                print(f"Deserialized object is not a numpy array, it's of type: {type(cluster_median)}")

            if USE_SUBSET_MEDIANS:
                # handles body lms subsets
                # print("handling body lms subsets in get_cluster_medians for these subset landmarks: ",sort.SUBSET_LANDMARKS)
                subset_cluster_median = []
                for i in range(len(cluster_median)):
                    # print("i: ",i)
                    if i in SUBSET_LANDMARKS:
                        # print("adding i: ",i)
                        subset_cluster_median.append(cluster_median[i])
                cluster_median = subset_cluster_median
                print(f"subset cluster median {cluster_id}: ",cluster_median)
            median_dict[cluster_id] = cluster_median
        # print("median dict: ",median_dict)
        self.CLUSTER_MEDIANS = median_dict
        return median_dict 

    def get_meta_cluster_dict(self, session, ClustersMetaClusters):
        print("getting cluster medians")
        # Create a SQLAlchemy select statement
        select_query = select(ClustersMetaClusters.cluster_id, ClustersMetaClusters.meta_cluster_id)

        # Execute the query using your SQLAlchemy session
        results = session.execute(select_query)
        meta_cluster_dict = {}

        # Process the results as needed
        # print(f'Found {len(results)} clusters with medians.')
        # print("results is a sqlalchemy.engine.result.ChunkedIteratorResult object at 0x3233d5400. this is how many: ",len(results))
        for row in results:
            # print(row)
            cluster_id, meta_cluster_id = row
            meta_cluster_dict[cluster_id] = meta_cluster_id

        return meta_cluster_dict 


    def set_table_cluster_type(self, META):
        self.META = META
            # set cluster_table_type for ArmsPoses3D, so it pulls from BodyPoses3D table
        # this allows the ArmsPoses3D value to set the Dict and subset landmarks.
        if self.CLUSTER_TYPE == "ArmsPoses3D":
            table_cluster_type = self.CLUSTER_TYPE
        else:
            if self.META:table_cluster_type = "BodyPoses3D"
            else:table_cluster_type = self.CLUSTER_TYPE
        return table_cluster_type
    
    def construct_table_classes(self, table_cluster_type):
        # handle the table objects based on cl.CLUSTER_TYPE
        ClustersTable_name = table_cluster_type
        ImagesClustersTable_name = "Images"+table_cluster_type
        MetaClustersTable_name = "Meta"+table_cluster_type
        ClustersMetaClustersTable_name = "Clusters"+MetaClustersTable_name

        print("ClustersTable_name: ",ClustersTable_name, " ImagesClustersTable_name: ",ImagesClustersTable_name, " MetaClustersTable_name: ",MetaClustersTable_name, " ClustersMetaClustersTable_name: ",ClustersMetaClustersTable_name)

        class Clusters(Base):
            # this doubles as MetaClusters
            __tablename__ = ClustersTable_name

            cluster_id = Column(Integer, primary_key=True, autoincrement=True)
            cluster_median = Column(BLOB)

        class ImagesClusters(Base):
            __tablename__ = ImagesClustersTable_name

            image_id = Column(Integer, ForeignKey(Images.image_id, ondelete="CASCADE"), primary_key=True)
            cluster_id = Column(Integer, ForeignKey(f'{ClustersTable_name}.cluster_id', ondelete="CASCADE"))
            cluster_dist = Column(Float)

        class MetaClusters(Base):
            __tablename__ = MetaClustersTable_name

            cluster_id = Column(Integer, primary_key=True, autoincrement=True)
            cluster_median = Column(BLOB)

        class ClustersMetaClusters(Base):
            __tablename__ = ClustersMetaClustersTable_name
            # cluster_id is pkey, and meta_cluster_id will appear multiple times for multiple clusters
            cluster_id = Column(Integer, ForeignKey(f'{ClustersTable_name}.cluster_id', ondelete="CASCADE"), primary_key=True)
            meta_cluster_id = Column(Integer, ForeignKey(f'{MetaClustersTable_name}.cluster_id', ondelete="CASCADE"))
            cluster_dist = Column(Float)
        return Clusters, ImagesClusters, MetaClusters, ClustersMetaClusters
    
    def set_cluster_metacluster(self, Clusters, ImagesClusters, MetaClusters, ClustersMetaClusters):
        if self.META:
            this_Cluster = MetaClusters
            this_CrosswalkClusters = ClustersMetaClusters
        else:
            this_Cluster = Clusters
            this_CrosswalkClusters = ImagesClusters
        return this_Cluster, this_CrosswalkClusters

    def prep_pose_clusters_enc(self, enc1, median_dict=None):
        # print("prepping pose clusters for enc1: ", enc1, " with median_dict: ", median_dict, " and CLUSTER_MEDIANS: ", self.CLUSTER_MEDIANS)
        if median_dict is None and self.CLUSTER_MEDIANS is not None:
            median_dict = self.CLUSTER_MEDIANS
        # print("current image enc1", enc1)  
        enc1 = np.array(enc1)
        
        this_dist_dict = {}
        for cluster_id in median_dict:
            enc2 = median_dict[cluster_id]
            # print("cluster_id enc2: ", cluster_id,enc2)
            this_dist_dict[cluster_id] = np.linalg.norm(enc1 - enc2, axis=0)
        
        cluster_id, cluster_dist = min(this_dist_dict.items(), key=lambda x: x[1])

        # print(cluster_id)
        return cluster_id, cluster_dist

    # ==================== OBJECT-HAND RELATIONSHIP METHODS ====================

    def parse_bbox_norm(self, bbox_norm):
        """Parse bbox_norm from various formats to dict."""
        if bbox_norm is None:
            return None

        # Already parsed JSON object (SQLAlchemy JSON column can return dict directly)
        elif isinstance(bbox_norm, dict):
            return bbox_norm

        elif isinstance(bbox_norm, str):
            try:
                # Handle double-encoded JSON
                parsed = json.loads(bbox_norm)
                if isinstance(parsed, str):
                    parsed = json.loads(parsed)
                return parsed
            except:
                return None

        else:
            print(f"[parse_bbox_norm] Unsupported bbox_norm type: {type(bbox_norm)}")
            return None

    def detection_to_list(self, detection_dict):
        """Convert detection dict to 6-element list: [class_id, conf, top, left, right, bottom]."""
        if detection_dict is None:
            return None
        return [
            detection_dict['class_id'],
            detection_dict['conf'],
            detection_dict['top'],
            detection_dict['left'],
            detection_dict['right'],
            detection_dict['bottom']
        ]

    def detection_to_payload(self, detection_dict):
        """Convert detection dict to a compact payload for DataFrame/DB use."""
        if detection_dict is None:
            return None

        return {
            'detection_id': int(detection_dict['detection_id']),
            'class_id': float(detection_dict['class_id']),
            'conf': float(detection_dict['conf']),
            'top': float(detection_dict['top']),
            'left': float(detection_dict['left']),
            'right': float(detection_dict['right']),
            'bottom': float(detection_dict['bottom']),
        }

    def _normalize_class_id(self, class_id_value):
        """Normalize class_id value to int if possible."""
        if class_id_value is None:
            return None
        try:
            return int(float(class_id_value))
        except (TypeError, ValueError):
            return None

    def _get_detection_class_id(self, detection_dict):
        """Get stable class_id for filtering, preferring raw class ID if available."""
        class_id_value = detection_dict.get('class_id_raw', detection_dict.get('class_id'))
        return self._normalize_class_id(class_id_value)

    def _is_class_in(self, detection_dict, class_id_set):
        class_id_value = self._get_detection_class_id(detection_dict)
        return class_id_value in class_id_set if class_id_value is not None else False

    def _load_compatibility_matrix_from_csv(self, all_class_ids):
        if not os.path.exists(self.COMPATIBILITY_MATRIX_PATH):
            raise FileNotFoundError(
                f"Compatibility matrix CSV not found: {self.COMPATIBILITY_MATRIX_PATH}"
            )

        try:
            matrix_df = pd.read_csv(self.COMPATIBILITY_MATRIX_PATH)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to read compatibility matrix CSV: {self.COMPATIBILITY_MATRIX_PATH}"
            ) from exc

        required_cols = ['class_id', *self.COMPATIBILITY_SLOT_COLUMNS]
        missing_cols = [col for col in required_cols if col not in matrix_df.columns]
        if missing_cols:
            raise ValueError(
                "Compatibility matrix missing required columns: " + ", ".join(missing_cols)
            )

        matrix_df = matrix_df[required_cols].copy()
        matrix_df['class_id'] = pd.to_numeric(matrix_df['class_id'], errors='coerce')
        if matrix_df['class_id'].isna().any():
            raise ValueError("Compatibility matrix has non-numeric class_id values")
        matrix_df['class_id'] = matrix_df['class_id'].astype(int)

        if matrix_df['class_id'].duplicated().any():
            dup_ids = matrix_df[matrix_df['class_id'].duplicated()]['class_id'].tolist()
            raise ValueError(f"Compatibility matrix has duplicate class_id rows: {dup_ids[:10]}")

        unknown_ids = sorted(list(set(matrix_df['class_id'].tolist()) - set(all_class_ids)))
        if unknown_ids:
            raise ValueError(f"Compatibility matrix has unknown class_id values: {unknown_ids[:10]}")

        missing_ids = sorted(list(set(all_class_ids) - set(matrix_df['class_id'].tolist())))
        if missing_ids:
            raise ValueError(
                f"Compatibility matrix missing class_id rows: {missing_ids[:10]}"
            )

        for col in self.COMPATIBILITY_SLOT_COLUMNS:
            numeric_col = pd.to_numeric(matrix_df[col], errors='coerce')
            if numeric_col.isna().any():
                raise ValueError(f"Compatibility matrix column '{col}' has non-numeric values")
            if (~numeric_col.isin([0, 1, 2])).any():
                raise ValueError(f"Compatibility matrix column '{col}' must contain only 0, 1, or 2")
            matrix_df[col] = numeric_col.astype(int)

        lookup = {}
        for row in matrix_df.to_dict('records'):
            class_id = int(row['class_id'])
            lookup[class_id] = {
                slot: int(row.get(slot, 2))
                for slot in self.COMPATIBILITY_SLOT_COLUMNS
            }
        return lookup

    def _get_slot_key_for_allowlist_slot(self, slot_name):
        if slot_name in ('left_hand', 'right_hand', 'hand'):
            return 'hand'
        return slot_name

    def _get_compatibility_level(self, class_id_value, slot_name):
        slot_key = self._get_slot_key_for_allowlist_slot(slot_name)
        if slot_key not in self.COMPATIBILITY_SLOT_COLUMNS:
            return 2
        if class_id_value is None:
            return 0
        class_row = self.compatibility_matrix.get(int(class_id_value))
        if not class_row:
            return 2
        return int(class_row.get(slot_key, 2))

    def _compatibility_biased_score(self, det, slot_name, base_score=None):
        class_id_value = self._get_detection_class_id(det)
        level = self._get_compatibility_level(class_id_value, slot_name)
        if level <= 0:
            return None

        score = float(det['conf']) if base_score is None else float(base_score)
        score += float(self.COMPATIBILITY_SCORE_BIAS.get(level, 0.0))
        return score

    def _extract_landmark_visibility(self, landmark):
        if landmark is None:
            return None
        if hasattr(landmark, 'visibility'):
            try:
                return float(landmark.visibility)
            except Exception:
                return None
        if isinstance(landmark, dict) and 'visibility' in landmark:
            try:
                return float(landmark['visibility'])
            except Exception:
                return None
        return None

    def _assess_lower_body_visibility(self, body_landmarks_normalized):
        """Estimate lower-body visibility for occlusion-aware feet/waist gating."""
        if body_landmarks_normalized is None:
            return {'known': False, 'visible': False, 'score': None}

        landmarks = None
        if hasattr(body_landmarks_normalized, 'landmark'):
            landmarks = body_landmarks_normalized.landmark
        elif isinstance(body_landmarks_normalized, (list, tuple, np.ndarray)):
            landmarks = body_landmarks_normalized

        if landmarks is None:
            return {'known': False, 'visible': False, 'score': None}

        visibility_indices = [23, 24, 25, 26, 27, 28]
        vals = []
        for idx in visibility_indices:
            if idx >= len(landmarks):
                continue
            vis = self._extract_landmark_visibility(landmarks[idx])
            if vis is not None:
                vals.append(vis)

        if not vals:
            return {'known': False, 'visible': False, 'score': None}

        mean_vis = float(sum(vals) / len(vals))
        return {
            'known': True,
            'visible': mean_vis >= self.LOWER_BODY_VISIBILITY_MIN,
            'score': mean_vis,
        }

    def _record_unassigned_reason(self, slot_name, reason):
        if slot_name not in self._slot_unassigned_reason_counts:
            return
        reason_key = str(reason or 'unspecified')
        slot_counts = self._slot_unassigned_reason_counts[slot_name]
        slot_counts[reason_key] = int(slot_counts.get(reason_key, 0)) + 1

    def reset_unassigned_reason_counts(self):
        self._slot_unassigned_reason_counts = {'waist': {}, 'feet': {}}

    def get_unassigned_reason_counts(self):
        return {
            slot: {reason: int(count) for reason, count in reasons.items()}
            for slot, reasons in self._slot_unassigned_reason_counts.items()
        }

    def _passes_slot_allowlist(self, detection_dict, slot_name):
        """Compatibility-gated eligibility for a slot (0=reject, 1=de-emphasize, 2=prefer)."""
        class_id_value = self._get_detection_class_id(detection_dict)
        return self._get_compatibility_level(class_id_value, slot_name) > 0

    def _record_allowlist_reject(self, slot_name):
        """Increment allowlist reject counter for one slot."""
        if not self.USE_ALLOWLIST:
            return
        if slot_name in self._allowlist_reject_counts:
            self._allowlist_reject_counts[slot_name] += 1

    def reset_allowlist_reject_counts(self):
        """Reset per-batch allowlist reject counters."""
        self._allowlist_reject_counts = {slot: 0 for slot in self._allowlist_slots}

    def get_allowlist_reject_counts(self):
        """Return a copy of current allowlist reject counters."""
        return dict(self._allowlist_reject_counts)

    def reset_class_assignment_debug_counts(self):
        """Reset per-batch assignment counters for tracked class IDs."""
        self._class_assignment_debug_counts = {
            class_id: {
                'seen': 0,
                'assigned_unique': 0,
                'assigned_slot_hits': 0,
                'dropped': 0,
                'slot_hits': {slot: 0 for slot in self._class_assignment_slot_names},
            }
            for class_id in self.DEBUG_TRACKED_CLASS_IDS
        }

    def reset_class_pipeline_debug_counts(self):
        """Reset per-batch pipeline-stage counters for tracked class IDs."""
        self._class_pipeline_debug_counts = {
            class_id: {
                'query_hits': 0,
                'bbox_parse_ok': 0,
                'bbox_parse_fail': 0,
                'post_resolve': 0,
                'tie_blocked': 0,
                'non_hand_pool': 0,
                'final_assigned_unique': 0,
                'parse_fail_examples': [],
            }
            for class_id in self.DEBUG_TRACKED_CLASS_IDS
        }

    def get_class_assignment_debug_counts(self):
        """Return a copy of tracked class assignment counters."""
        snapshot = {}
        for class_id, stats in self._class_assignment_debug_counts.items():
            snapshot[class_id] = {
                'seen': int(stats['seen']),
                'assigned_unique': int(stats['assigned_unique']),
                'assigned_slot_hits': int(stats['assigned_slot_hits']),
                'dropped': int(stats['dropped']),
                'slot_hits': {slot: int(count) for slot, count in stats['slot_hits'].items()},
            }
        return snapshot

    def get_class_pipeline_debug_counts(self):
        """Return a copy of tracked class pipeline counters."""
        snapshot = {}
        for class_id, stats in self._class_pipeline_debug_counts.items():
            snapshot[class_id] = {
                'query_hits': int(stats['query_hits']),
                'bbox_parse_ok': int(stats['bbox_parse_ok']),
                'bbox_parse_fail': int(stats['bbox_parse_fail']),
                'post_resolve': int(stats['post_resolve']),
                'tie_blocked': int(stats['tie_blocked']),
                'non_hand_pool': int(stats['non_hand_pool']),
                'final_assigned_unique': int(stats['final_assigned_unique']),
                'parse_fail_examples': list(stats['parse_fail_examples']),
            }
        return snapshot

    def calc_iou(self, bbox1, bbox2):
        """Calculate Intersection over Union between two bboxes."""
        x_left = max(bbox1['left'], bbox2['left'])
        x_right = min(bbox1['right'], bbox2['right'])
        y_top = max(bbox1['top'], bbox2['top'])
        y_bottom = min(bbox1['bottom'], bbox2['bottom'])
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        area1 = (bbox1['right'] - bbox1['left']) * (bbox1['bottom'] - bbox1['top'])
        area2 = (bbox2['right'] - bbox2['left']) * (bbox2['bottom'] - bbox2['top'])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0

    def point_to_bbox_distance(self, point, bbox):
        """
        Calculate minimum distance from a point (x, y) to bbox edges.
        Returns 0 if point is inside bbox, otherwise returns distance to nearest edge.
        """
        px, py = point[0], point[1]  # x, y from knuckle coords
        
        # Check if inside bbox
        if bbox['left'] <= px <= bbox['right'] and bbox['top'] <= py <= bbox['bottom']:
            return 0.0
        
        # Calculate distance to each edge
        dx = max(bbox['left'] - px, 0, px - bbox['right'])
        dy = max(bbox['top'] - py, 0, py - bbox['bottom'])
        
        return (dx**2 + dy**2)**0.5

    def is_touching_hand(self, knuckle_pos, bbox):
        """Check if knuckle is within TOUCH_THRESHOLD of bbox."""
        if knuckle_pos == self.DEFAULT_HAND_POSITION:
            return False
        dist = self.point_to_bbox_distance(knuckle_pos, bbox)
        return dist <= self.TOUCH_THRESHOLD

    def is_top_face_object(self, bbox):
        """Check if object is on top of face (above nose, spans both sides)."""
        # Top of bbox is above nose (in this coord system, negative y is above)
        # AND bbox spans both sides of nose (has both positive and negative x)
        # AND object is not too wide (max width constraint)
        # AND object doesn't extend too far into bottom zone
        width = bbox['right'] - bbox['left']
        extends_into_bottom = max(0, bbox['bottom'])  # how far does it go into positive y
        
        return (bbox['top'] < self.CENTERLINE_Y and 
            bbox['left'] < self.CENTERLINE_X and 
            bbox['right'] > self.CENTERLINE_X and
                width <= self.MAX_FACE_WIDTH and
                extends_into_bottom <= self.MAX_FACE_VERT_EXTENSION)

    def is_bottom_face_object(self, bbox):
        """Check if object is on bottom of face (below nose, spans both sides)."""
        # Bottom of bbox is below nose (positive y)
        # AND bbox spans both sides of nose
        # AND object is not too wide (max width constraint)
        # AND object doesn't extend too far into top zone
        width = bbox['right'] - bbox['left']
        extends_into_top = max(0, -bbox['top'])  # how far does it go into negative y
        
        return (bbox['bottom'] > self.CENTERLINE_Y and 
            bbox['left'] < self.CENTERLINE_X and 
            bbox['right'] > self.CENTERLINE_X and
                width <= self.MAX_FACE_WIDTH and
                extends_into_top <= self.MAX_FACE_VERT_EXTENSION)

    def is_mouth_object(self, bbox):
        """Check if object is on mouth area with zero top-zone tolerance."""
        width = bbox['right'] - bbox['left']
        extends_into_top = max(0, -bbox['top'])

        return (bbox['bottom'] > self.CENTERLINE_Y and
            bbox['left'] < self.CENTERLINE_X and
            bbox['right'] > self.CENTERLINE_X and
                width <= self.MAX_FACE_WIDTH and
            extends_into_top <= self.MOUTH_MAX_TOP_EXTENSION)

    def is_tie_neck_object(self, bbox):
        """Check if tie bbox matches a centered neck/chest profile."""
        width = bbox['right'] - bbox['left']
        height = bbox['bottom'] - bbox['top']
        center_x = (bbox['left'] + bbox['right']) / 2.0

        if width <= 0 or height <= 0:
            return False

        return (
            bbox['left'] < 0
            and bbox['right'] > 0
            and abs(center_x) <= self.TIE_NECK_CENTER_X_TOL
            and width <= self.TIE_NECK_MAX_WIDTH
            and bbox['top'] <= self.TIE_NECK_TOP_MAX
            and bbox['bottom'] >= self.TIE_NECK_BOTTOM_MIN
            and bbox['bottom'] <= self.TIE_NECK_BOTTOM_MAX
            and height >= self.TIE_NECK_MIN_HEIGHT
            and (height / width) >= self.TIE_NECK_MIN_ASPECT_RATIO
        )

    def is_covid_mask_object(self, bbox):
        """Check if bbox looks like a covid mask, including pulled-below-chin cases."""
        width = bbox['right'] - bbox['left']
        return (
            bbox['left'] < self.CENTERLINE_X and
            bbox['right'] > self.CENTERLINE_X and
            width <= self.MAX_FACE_WIDTH and
            bbox['top'] >= self.COVID_MASK_TOP_MIN and
            bbox['top'] <= self.COVID_MASK_TOP_MAX and
            bbox['bottom'] >= self.COVID_MASK_BOTTOM_MIN and
            bbox['bottom'] <= self.COVID_MASK_BOTTOM_MAX
        )

    def _bbox_intersect_fraction(self, bbox, x_min, x_max, y_top, y_bottom):
        """Return the fraction of bbox area that overlaps with the given rectangle.

        Used to guard lower-body slot assignments when a minimum zone-intersection
        ratio is required (Rule Spec v1.0 classes 108-109).
        """
        ix_left = max(bbox['left'], x_min)
        ix_right = min(bbox['right'], x_max)
        iy_top = max(bbox['top'], y_top)
        iy_bottom = min(bbox['bottom'], y_bottom)
        if ix_right <= ix_left or iy_bottom <= iy_top:
            return 0.0
        intersection = (ix_right - ix_left) * (iy_bottom - iy_top)
        bbox_area = (bbox['right'] - bbox['left']) * (bbox['bottom'] - bbox['top'])
        if bbox_area <= 0:
            return 0.0
        return intersection / bbox_area

    def _classify_mask_intent(self, bbox, left_knuckle, right_knuckle):
        """Classify a class-110 mask detection as worn_on_face, held_in_hand, or ambiguous.

        Returns (intent_str, S_worn, S_held).
        Rule Spec v1.0 -- mask intent scoring.
        """
        cx = (bbox['left'] + bbox['right']) / 2.0
        cy = (bbox['top'] + bbox['bottom']) / 2.0

        # Mouth geometry: how many of the 4 covid-mask constraints are satisfied
        mask_checks = [
            bbox['top'] >= self.COVID_MASK_TOP_MIN,
            bbox['top'] <= self.COVID_MASK_TOP_MAX,
            bbox['bottom'] >= self.COVID_MASK_BOTTOM_MIN,
            bbox['bottom'] <= self.COVID_MASK_BOTTOM_MAX,
        ]
        s_mouthgeom = sum(mask_checks) / 4.0

        # Face-anchor: centered near nose-mouth region
        cx_scale = 0.35
        cy_target = 0.2
        cy_scale = 0.9
        s_faceanchor = (
            max(0.0, 1.0 - abs(cx - self.CENTERLINE_X) / cx_scale)
            * max(0.0, 1.0 - abs(cy - cy_target) / cy_scale)
        )

        # Centerline proximity
        s_centerline = max(0.0, 1.0 - abs(cx - self.CENTERLINE_X) / cx_scale)

        # Hand proximity (0 = far, 1 = touching)
        hand_dists = []
        if left_knuckle != self.DEFAULT_HAND_POSITION:
            hand_dists.append(self.point_to_bbox_distance(left_knuckle, bbox))
        if right_knuckle != self.DEFAULT_HAND_POSITION:
            hand_dists.append(self.point_to_bbox_distance(right_knuckle, bbox))
        if hand_dists:
            d_min = min(hand_dists)
            s_handprox = max(0.0, 1.0 - d_min / (2.0 * self.TOUCH_THRESHOLD))
        else:
            s_handprox = 0.0

        # Dual-hand indicator
        s_dualhand = 1.0 if (
            len(hand_dists) == 2
            and all(d <= self.TOUCH_THRESHOLD for d in hand_dists)
        ) else 0.0

        # Off-face: mask center is outside the expected face zone and geometry fails
        s_offface = 1.0 if (
            s_mouthgeom < 0.5
            and (cy < self.COVID_MASK_TOP_MIN or cy > self.COVID_MASK_BOTTOM_MAX)
        ) else 0.0

        S_worn = (
            self.W_MASK_MOUTH_GEOM * s_mouthgeom
            + self.W_MASK_FACE_ANCHOR * s_faceanchor
            + self.W_MASK_CENTERLINE * s_centerline
            - self.W_MASK_HAND_PROX * s_handprox
        )
        S_held = (
            self.W_MASK_HAND_PROX * s_handprox
            + self.W_MASK_DUAL_HAND * s_dualhand
            + self.W_MASK_OFF_FACE * s_offface
            - self.W_MASK_MOUTH_GEOM * s_mouthgeom
        )

        if S_worn >= self.MASK_INTENT_WORN_MIN and S_worn >= S_held + self.MASK_INTENT_MARGIN:
            return 'worn_on_face', S_worn, S_held
        if S_held >= self.MASK_INTENT_HELD_MIN and S_held >= S_worn + self.MASK_INTENT_MARGIN:
            return 'held_in_hand', S_worn, S_held
        return 'ambiguous', S_worn, S_held

    def is_under_eye_object(self, bbox):
        """Check if bbox falls in an under-eye treatment zone."""
        return self._bbox_intersects_rect(
            bbox,
            self.LEFT_EYE_X_MIN,
            self.RIGHT_EYE_X_MAX,
            self.UNDER_EYE_ZONE_TOP,
            self.UNDER_EYE_ZONE_BOTTOM,
        )

    def is_eye_cover_object(self, bbox):
        """Check if bbox intersects a broad eye-cover zone used by masks and slices."""
        return self._bbox_intersects_rect(
            bbox,
            self.LEFT_EYE_X_MIN,
            self.RIGHT_EYE_X_MAX,
            self.EYE_COVER_ZONE_TOP,
            self.EYE_COVER_ZONE_BOTTOM,
        )

    def is_forehead_object(self, bbox):
        """Check if bbox intersects a forehead / pushed-up mask zone."""
        return self._bbox_intersects_rect(
            bbox,
            self.FOREHEAD_X_MIN,
            self.FOREHEAD_X_MAX,
            self.FOREHEAD_ZONE_TOP,
            self.FOREHEAD_ZONE_BOTTOM,
        )

    def _bbox_intersects_rect(self, bbox, x_min, x_max, y_top, y_bottom):
        """Check whether bbox intersects a rectangular zone."""
        return not (
            bbox['right'] < x_min
            or bbox['left'] > x_max
            or bbox['bottom'] < y_top
            or bbox['top'] > y_bottom
        )

    def is_full_face_mask_object(self, bbox):
        """Check if bbox covers both eyes+mouth region (full-face sheet mask style)."""
        width = bbox['right'] - bbox['left']
        return (
            bbox['left'] < self.FULL_FACE_MASK_X_MIN
            and bbox['right'] > self.FULL_FACE_MASK_X_MAX
            and bbox['top'] <= self.FULL_FACE_MASK_TOP_MAX
            and bbox['bottom'] >= self.FULL_FACE_MASK_BOTTOM_MIN
            and width <= self.MAX_FACE_WIDTH
        )

    def is_left_eye_object(self, bbox):
        """Check if bbox intersects the left-eye zone."""
        return self._bbox_intersects_rect(
            bbox,
            self.LEFT_EYE_X_MIN,
            self.LEFT_EYE_X_MAX,
            self.EYE_ZONE_TOP,
            self.EYE_ZONE_BOTTOM,
        )

    def is_right_eye_object(self, bbox):
        """Check if bbox intersects the right-eye zone."""
        return self._bbox_intersects_rect(
            bbox,
            self.RIGHT_EYE_X_MIN,
            self.RIGHT_EYE_X_MAX,
            self.EYE_ZONE_TOP,
            self.EYE_ZONE_BOTTOM,
        )

    def is_waist_object(self, bbox):
        """Check if bbox intersects a broad waist/seat interaction zone."""
        return self._bbox_intersects_rect(
            bbox,
            self.WAIST_X_MIN,
            self.WAIST_X_MAX,
            self.WAIST_ZONE_TOP,
            self.WAIST_ZONE_BOTTOM,
        )

    def is_feet_object(self, bbox):
        """Check if bbox intersects a broad lower-body/feet zone."""
        return self._bbox_intersects_rect(
            bbox,
            self.FEET_X_MIN,
            self.FEET_X_MAX,
            self.FEET_ZONE_TOP,
            self.FEET_ZONE_BOTTOM,
        )

    def _extract_xy_from_landmark(self, landmark):
        """Extract (x, y) from a landmark in object/dict/list form."""
        if landmark is None:
            return None

        if hasattr(landmark, 'x') and hasattr(landmark, 'y'):
            return [float(landmark.x), float(landmark.y)]

        if isinstance(landmark, dict):
            if 'x' in landmark and 'y' in landmark:
                return [float(landmark['x']), float(landmark['y'])]
            return None

        if isinstance(landmark, (list, tuple, np.ndarray)) and len(landmark) >= 2:
            return [float(landmark[0]), float(landmark[1])]

        return None

    def extract_shoulder_points(self, body_landmarks_normalized):
        """Extract left/right shoulder points (landmarks 11/12) as [x, y]."""
        if body_landmarks_normalized is None:
            return None, None

        try:
            landmarks = None
            if hasattr(body_landmarks_normalized, 'landmark'):
                landmarks = body_landmarks_normalized.landmark
            elif isinstance(body_landmarks_normalized, (list, tuple, np.ndarray)):
                landmarks = body_landmarks_normalized

            if landmarks is None or len(landmarks) <= 12:
                return None, None

            left_shoulder = self._extract_xy_from_landmark(landmarks[11])
            right_shoulder = self._extract_xy_from_landmark(landmarks[12])
            return left_shoulder, right_shoulder
        except Exception:
            return None, None

    def is_shoulder_object(self, bbox, left_shoulder, right_shoulder):
        """
        Check if object crosses the shoulder band.
        Shoulder band is the line from lm11 to lm12, extended 1.0 unit lower.
        """
        if left_shoulder is None or right_shoulder is None:
            return False

        x1, y1 = left_shoulder
        x2, y2 = right_shoulder

        shoulder_x_min = min(x1, x2)
        shoulder_x_max = max(x1, x2)

        if bbox['right'] < shoulder_x_min or bbox['left'] > shoulder_x_max:
            return False

        overlap_left = max(bbox['left'], shoulder_x_min)
        overlap_right = min(bbox['right'], shoulder_x_max)
        if overlap_right < overlap_left:
            return False

        if abs(x2 - x1) < 1e-9:
            shoulder_y_min = min(y1, y2)
            shoulder_y_max = max(y1, y2)
        else:
            def y_on_shoulder_line(x_val):
                t = (x_val - x1) / (x2 - x1)
                return y1 + t * (y2 - y1)

            y_left = y_on_shoulder_line(overlap_left)
            y_right = y_on_shoulder_line(overlap_right)
            y_mid = y_on_shoulder_line((overlap_left + overlap_right) / 2.0)
            shoulder_y_min = min(y_left, y_right, y_mid)
            shoulder_y_max = max(y_left, y_right, y_mid)

        band_top = shoulder_y_min
        band_bottom = shoulder_y_max + self.SHOULDER_BAND_EXTENSION

        return not (bbox['bottom'] < band_top or bbox['top'] > band_bottom)

    def resolve_overlapping_detections(self, detections):
        """
        Resolve overlapping detections by keeping the best one based on confidence rules.
        Returns filtered list of detections.
        """
        if len(detections) <= 1:
            return detections
        
        filtered = []
        used_indices = set()
        
        for i, det1 in enumerate(detections):
            if i in used_indices:
                continue
                
            best_det = det1
            
            for j, det2 in enumerate(detections):
                if j <= i or j in used_indices:
                    continue
                    
                iou = self.calc_iou(det1['bbox'], det2['bbox'])
                
                if iou >= self.OVERLAP_IOU_THRESHOLD:
                    # Overlapping detections - resolve based on confidence
                    conf1, conf2 = det1['conf'], det2['conf']
                    conf_diff = abs(conf1 - conf2)
                    
                    conf_diff_threshold_scaled_to_iou = self.CONFIDENCE_DIFF_THRESHOLD - (self.CONFIDENCE_DIFF_THRESHOLD * iou)
                    if conf1 >= self.HIGH_CONFIDENCE_THRESHOLD or conf2 >= self.HIGH_CONFIDENCE_THRESHOLD or conf_diff >= conf_diff_threshold_scaled_to_iou:
                        winner = det1 if conf1 >= conf2 else det2
                        loser = det2 if conf1 >= conf2 else det1
                        if self.VERBOSE:
                            print(f"  ✅ OVERLAP RESOLVED: Chose class {winner['class_id']} (conf={winner['conf']:.2f}) over class {loser['class_id']} (conf={loser['conf']:.2f}), IoU={iou:.2f}")
                        best_det = winner
                        used_indices.add(j)
                    
                    if det1['class_id'] == det2['class_id']:
                        # same class...
                        if iou >= self.OVERLAP_IOU_THRESHOLD*1.5:
                            # take the union of the boxes
                            new_bbox = {
                                'top': min(det1['bbox']['top'], det2['bbox']['top']),
                                'left': min(det1['bbox']['left'], det2['bbox']['left']),
                                'right': max(det1['bbox']['right'], det2['bbox']['right']),
                                'bottom': max(det1['bbox']['bottom'], det2['bbox']['bottom']),
                            }
                            best_det['bbox'] = new_bbox
                            print(f"  🔄 MERGED SAME CLASS OVERLAP: class {det1['class_id']} (conf={det1['conf']:.2f}) and class {det2['class_id']} (conf={det2['conf']:.2f}), IoU={iou:.2f} - merged bbox")
                        else:
                            # keep higher confidence
                            winner = det1 if conf1 >= conf2 else det2
                            loser = det2 if conf1 >= conf2 else det1
                            print(f"  ⚠️ SAME CLASS OVERLAP RESOLVED: Chose class {winner['class_id']} (conf={winner['conf']:.2f}) over class {loser['class_id']} (conf={loser['conf']:.2f}), IoU={iou:.2f}, conf_diff={conf_diff:.2f}")
                            best_det = winner
                        used_indices.add(j)

                    elif conf1 >= self.MIN_DETECTION_CONFIDENCE*1.5 or conf2 >= self.MIN_DETECTION_CONFIDENCE*1.5:
                        # both moderate confidence, keep higher
                        winner = det1 if conf1 >= conf2 else det2
                        loser = det2 if conf1 >= conf2 else det1
                        if self.VERBOSE:
                            print(f"  ❌ HIGH CONF RESOLVED: Chose class {winner['class_id']} (conf={winner['conf']:.2f}) over class {loser['class_id']} (conf={loser['conf']:.2f}), IoU={iou:.2f}")
                        best_det = winner
                        used_indices.add(j)

                    else:
                        # Cannot determine - alert and discard both
                        if self.VERBOSE:
                            print(f"  🚨 OVERLAP UNRESOLVED - DISCARDING: classes {det1['class_id']} (conf={conf1:.2f}) and {det2['class_id']} (conf={conf2:.2f}), IoU={iou:.2f} - keeping both")
            

            filtered.append(best_det)
            used_indices.add(i)
        
        return filtered

    def weight_detection_for_clustering(self, detections):
        """
        Apply weighting to detections features based on class_id for clustering.
        Note: If USE_FEATURE_STANDARDIZATION=True, this weight is applied BEFORE standardization,
        then features are standardized, then FEATURE_WEIGHTS are applied after.
        If USE_FEATURE_STANDARDIZATION=False, only CLASS_ID_WEIGHT is used (legacy behavior).
        """
        if not self.USE_FEATURE_STANDARDIZATION:
            # Legacy behavior: simple multiplication
            for det in detections:
                det['class_id'] *= self.CLASS_ID_WEIGHT
        else:
            # New behavior: CLASS_ID_WEIGHT is incorporated into FEATURE_WEIGHTS['class_id']
            # Don't multiply here - let standardization handle it
            pass
        return detections
    
    def classify_object_hand_relationships(
        self,
        detections,
        left_knuckle,
        right_knuckle,
        left_shoulder=None,
        right_shoulder=None,
        body_landmarks_normalized=None,
    ):
        """
        Classify each detection based on its relationship to hands and face.
        Returns dict with keys: left_hand_object, right_hand_object,
                               top_face_object, left_eye_object, right_eye_object,
                               mouth_object, shoulder_object, waist_object, feet_object
        Each value is the detection dict or None.
        """

        # weight the class_ids for better clustering separation of classes
        detections = self.weight_detection_for_clustering(detections)
        
        results = {
            'left_hand_object': None,
            'right_hand_object': None,
            'top_face_object': None,
            'left_eye_object': None,
            'right_eye_object': None,
            'mouth_object': None,
            'shoulder_object': None,
            'waist_object': None,
            'feet_object': None,
        }
        
        if not detections:
            return results
        
        # First, resolve overlapping detections
        detections = self.resolve_overlapping_detections(detections)

        for det in detections:
            class_id = self._get_detection_class_id(det)
            if class_id in self._class_pipeline_debug_counts:
                self._class_pipeline_debug_counts[class_id]['post_resolve'] += 1

        tie_locked_slots = set()
        tie_blocked_detection_ids = set()
        lower_body_visibility = self._assess_lower_body_visibility(body_landmarks_normalized)

        def assign_slot_if_preferred(slot_name, det):
            slot_key = 'hand' if slot_name in ('left_hand_object', 'right_hand_object') else slot_name.replace('_object', '')
            det_score = self._compatibility_biased_score(det, slot_key)
            if det_score is None:
                return

            if results[slot_name] is None:
                results[slot_name] = det
                return

            current_score = self._compatibility_biased_score(results[slot_name], slot_key)
            if current_score is None or det_score > current_score or (
                det_score == current_score and det['conf'] > results[slot_name]['conf']
            ):
                results[slot_name] = det

        tie_detections = [
            det for det in detections
            if self._get_detection_class_id(det) == self.TIE_CLASS_ID
        ]
        tie_detections.sort(key=lambda det: det.get('conf', 0.0), reverse=True)

        for tie_det in tie_detections:
            tie_bbox = tie_det['bbox']
            tie_detection_id = tie_det['detection_id']

            hand_allowed = self._passes_slot_allowlist(tie_det, 'hand')
            shoulder_allowed = self._passes_slot_allowlist(tie_det, 'shoulder')
            mouth_allowed = self._passes_slot_allowlist(tie_det, 'mouth')

            if not hand_allowed:
                self._record_allowlist_reject('hand')
            if not shoulder_allowed:
                self._record_allowlist_reject('shoulder')
            if not mouth_allowed:
                self._record_allowlist_reject('mouth')

            neck_match = shoulder_allowed and self.is_shoulder_object(tie_bbox, left_shoulder, right_shoulder)
            left_touching = hand_allowed and self.is_touching_hand(left_knuckle, tie_bbox)
            right_touching = hand_allowed and self.is_touching_hand(right_knuckle, tie_bbox)
            mouth_match = mouth_allowed and (
                self.is_mouth_object(tie_bbox) or self.is_tie_neck_object(tie_bbox)
            )

            if neck_match:
                assign_slot_if_preferred('shoulder_object', tie_det)
                tie_locked_slots.add('shoulder_object')
                tie_blocked_detection_ids.add(tie_detection_id)
                if self.VERBOSE:
                    print(f"  [TIE] detection_id={tie_detection_id} -> shoulder_object (neck priority)")
                continue

            if left_touching and right_touching:
                assign_slot_if_preferred('left_hand_object', tie_det)
                assign_slot_if_preferred('right_hand_object', tie_det)
                tie_locked_slots.update(['left_hand_object', 'right_hand_object'])
                tie_blocked_detection_ids.add(tie_detection_id)
                if self.VERBOSE:
                    print(f"  [TIE] detection_id={tie_detection_id} -> both hands")
                continue

            if right_touching:
                assign_slot_if_preferred('right_hand_object', tie_det)
                tie_locked_slots.add('right_hand_object')
                tie_blocked_detection_ids.add(tie_detection_id)
                if self.VERBOSE:
                    print(f"  [TIE] detection_id={tie_detection_id} -> right_hand_object")
                continue

            if left_touching:
                assign_slot_if_preferred('left_hand_object', tie_det)
                tie_locked_slots.add('left_hand_object')
                tie_blocked_detection_ids.add(tie_detection_id)
                if self.VERBOSE:
                    print(f"  [TIE] detection_id={tie_detection_id} -> left_hand_object")
                continue

            if mouth_match:
                assign_slot_if_preferred('mouth_object', tie_det)
                tie_locked_slots.add('mouth_object')
                tie_blocked_detection_ids.add(tie_detection_id)
                if self.VERBOSE:
                    print(f"  [TIE] detection_id={tie_detection_id} -> mouth_object (neck fallback)")
                continue

            tie_blocked_detection_ids.add(tie_detection_id)
            if self.VERBOSE:
                print(f"  [TIE] detection_id={tie_detection_id} -> no assignment")

        tie_blocked_ids_set = set(tie_blocked_detection_ids)
        for det in detections:
            class_id = self._get_detection_class_id(det)
            if class_id not in self._class_pipeline_debug_counts:
                continue
            if int(det.get('detection_id')) in tie_blocked_ids_set:
                self._class_pipeline_debug_counts[class_id]['tie_blocked'] += 1

        # 1. Find best object for each hand independently (same object may be assigned to both)
        def best_object_for_hand(knuckle, existing_hand_object=None):
            if existing_hand_object is not None:
                return existing_hand_object
            if knuckle == self.DEFAULT_HAND_POSITION:
                return None

            touching_candidates = []
            nearby_candidates = []

            for det in detections:
                if det['detection_id'] in tie_blocked_detection_ids:
                    continue
                if not self._passes_slot_allowlist(det, 'hand'):
                    self._record_allowlist_reject('hand')
                    continue
                dist = self.point_to_bbox_distance(knuckle, det['bbox'])
                compat_score = self._compatibility_biased_score(det, 'hand')
                if compat_score is None:
                    continue
                if dist <= self.TOUCH_THRESHOLD:
                    touching_candidates.append((dist, -compat_score, det))
                elif dist <= self.TOUCH_THRESHOLD * 2:
                    nearby_candidates.append((dist, -compat_score, det))

            if touching_candidates:
                touching_candidates.sort(key=lambda item: item[0])
                return touching_candidates[0][2]

            if nearby_candidates:
                nearby_candidates.sort(key=lambda item: item[0])
                return nearby_candidates[0][2]

            return None

        results['left_hand_object'] = best_object_for_hand(left_knuckle, results['left_hand_object'])
        results['right_hand_object'] = best_object_for_hand(right_knuckle, results['right_hand_object'])

        # Hand-assigned objects are excluded from all other zones.
        # Exception: class 110 (COVID mask) detections classified as worn_on_face
        # remain eligible for mouth_object even when also assigned to a hand slot.
        hand_detection_ids = set()
        for hand_key in ['left_hand_object', 'right_hand_object']:
            hand_det = results[hand_key]
            if hand_det is not None:
                hand_detection_ids.add(hand_det['detection_id'])

        # Pre-compute mask intent for all class 110 detections.
        mask_110_intents = {}
        for det in detections:
            if self._get_detection_class_id(det) in self.COVID_MASK_CLASSES:
                intent, _, _ = self._classify_mask_intent(det['bbox'], left_knuckle, right_knuckle)
                mask_110_intents[det['detection_id']] = intent

        # Worn-on-face masks are exempt from the hand exclusion so they
        # can still compete for mouth_object in stage 3.
        mask_worn_exception_ids = {
            did for did, intent in mask_110_intents.items()
            if intent == 'worn_on_face' and did in hand_detection_ids
        }

        non_hand_detections = [
            det for det in detections
            if (
                det['detection_id'] not in hand_detection_ids
                or det['detection_id'] in mask_worn_exception_ids
            )
            and det['detection_id'] not in tie_blocked_detection_ids
        ]

        def normalized_distance_to_zone_center(bbox, slot_name):
            """Lightweight proximity proxy for lower-body slots.

            Smaller is better; used only as a tie-break style guardrail for small handhelds.
            """
            cx = (bbox['left'] + bbox['right']) / 2.0
            cy = (bbox['top'] + bbox['bottom']) / 2.0

            if slot_name == 'waist':
                zone_y_center = (self.WAIST_ZONE_TOP + self.WAIST_ZONE_BOTTOM) / 2.0
                x_scale = max(abs(self.WAIST_X_MIN), abs(self.WAIST_X_MAX), 1e-6)
                y_scale = max((self.WAIST_ZONE_BOTTOM - self.WAIST_ZONE_TOP) / 2.0, 1e-6)
            elif slot_name == 'feet':
                zone_y_center = (self.FEET_ZONE_TOP + self.FEET_ZONE_BOTTOM) / 2.0
                x_scale = max(abs(self.FEET_X_MIN), abs(self.FEET_X_MAX), 1e-6)
                y_scale = max((self.FEET_ZONE_BOTTOM - self.FEET_ZONE_TOP) / 2.0, 1e-6)
            else:
                return 0.0

            dx = cx / x_scale
            dy = (cy - zone_y_center) / y_scale
            return (dx * dx + dy * dy) ** 0.5

        def lower_body_candidate_score(det, slot_name):
            """Score lower-body candidate with mild penalties for small handheld classes.

            Penalty is removed only when lower-body geometry is clearly better than hand proximity.
            """
            score = self._compatibility_biased_score(det, slot_name, base_score=det['conf'])
            if score is None:
                return None, 'incompatible_by_matrix'

            class_id = self._get_detection_class_id(det)
            if class_id not in self.HANDHELD_LIKE_CLASSES:
                return score, None

            bbox = det['bbox']
            hand_dists = []
            if left_knuckle != self.DEFAULT_HAND_POSITION:
                hand_dists.append(self.point_to_bbox_distance(left_knuckle, bbox))
            if right_knuckle != self.DEFAULT_HAND_POSITION:
                hand_dists.append(self.point_to_bbox_distance(right_knuckle, bbox))

            if lower_body_visibility['known'] and not lower_body_visibility['visible']:
                score -= self.LOWER_BODY_OCCLUSION_PENALTY

            # If no hand landmarks are available, keep penalty to avoid over-pulling to lower body.
            if not hand_dists:
                score -= self.SMALL_HANDHELD_LOWER_BODY_CONF_PENALTY
                if class_id == 67:
                    score -= self.CLASS67_EXTRA_LOWER_BODY_PENALTY.get(slot_name, 0.0)
                if score < self.HANDHELD_LIKE_LOWER_BODY_MIN_SCORE:
                    reason = 'weak_lower_body_evidence_no_hand_landmarks'
                    if lower_body_visibility['known'] and not lower_body_visibility['visible']:
                        reason = 'lower_body_occluded_and_no_hand_landmarks'
                    return None, reason
                return score, None

            best_hand_dist = min(hand_dists)
            lower_body_dist = normalized_distance_to_zone_center(bbox, slot_name)

            # Waive penalty only when lower-body fit is meaningfully better than hand proximity.
            distance_margin = best_hand_dist - lower_body_dist
            if distance_margin < self.SMALL_HANDHELD_LOWER_BODY_DISTANCE_MARGIN:
                score -= self.SMALL_HANDHELD_LOWER_BODY_CONF_PENALTY
                if class_id == 67:
                    score -= self.CLASS67_EXTRA_LOWER_BODY_PENALTY.get(slot_name, 0.0)

            if score < self.HANDHELD_LIKE_LOWER_BODY_MIN_SCORE:
                reason = 'weak_lower_body_evidence'
                if lower_body_visibility['known'] and not lower_body_visibility['visible']:
                    reason = 'lower_body_occluded'
                return None, reason

            return score, None

        for det in non_hand_detections:
            class_id = self._get_detection_class_id(det)
            if class_id in self._class_pipeline_debug_counts:
                self._class_pipeline_debug_counts[class_id]['non_hand_pool'] += 1

        # 2. Eye assignments (same object may map to both eyes, e.g., eyeglasses)
        for det in non_hand_detections:
            bbox = det['bbox']

            if self._is_class_in(det, self.HAND_ONLY_CLASSES | self.COVID_MASK_CLASSES | self.FULL_FACE_TOP_BIASED_CLASSES):
                continue

            left_eye_allowed = self._passes_slot_allowlist(det, 'left_eye')
            right_eye_allowed = self._passes_slot_allowlist(det, 'right_eye')

            if not left_eye_allowed:
                self._record_allowlist_reject('left_eye')
            if not right_eye_allowed:
                self._record_allowlist_reject('right_eye')

            if not left_eye_allowed and not right_eye_allowed:
                continue

            if self.is_full_face_mask_object(bbox):
                continue

            if self._is_class_in(det, self.UNDER_EYE_CLASSES):
                if left_eye_allowed and bbox['right'] > 0 and bbox['left'] < 0 and self.is_under_eye_object(bbox):
                    if results['left_eye_object'] is None:
                        results['left_eye_object'] = det
                    elif det['conf'] > results['left_eye_object']['conf']:
                        results['left_eye_object'] = det
                    if results['right_eye_object'] is None:
                        results['right_eye_object'] = det
                    elif det['conf'] > results['right_eye_object']['conf']:
                        results['right_eye_object'] = det
                else:
                    if left_eye_allowed and self._bbox_intersects_rect(
                        bbox,
                        self.LEFT_EYE_X_MIN,
                        self.LEFT_EYE_X_MAX,
                        self.UNDER_EYE_ZONE_TOP,
                        self.UNDER_EYE_ZONE_BOTTOM,
                    ):
                        if results['left_eye_object'] is None:
                            results['left_eye_object'] = det
                        elif det['conf'] > results['left_eye_object']['conf']:
                            results['left_eye_object'] = det
                    if right_eye_allowed and self._bbox_intersects_rect(
                        bbox,
                        self.RIGHT_EYE_X_MIN,
                        self.RIGHT_EYE_X_MAX,
                        self.UNDER_EYE_ZONE_TOP,
                        self.UNDER_EYE_ZONE_BOTTOM,
                    ):
                        if results['right_eye_object'] is None:
                            results['right_eye_object'] = det
                        elif det['conf'] > results['right_eye_object']['conf']:
                            results['right_eye_object'] = det
                continue

            if self._is_class_in(det, self.EYE_OR_FOREHEAD_CLASSES | self.EYE_ONLY_CLASSES | self.HAND_OR_EYE_CLASSES):
                if left_eye_allowed and self._bbox_intersects_rect(
                    bbox,
                    self.LEFT_EYE_X_MIN,
                    self.LEFT_EYE_X_MAX,
                    self.EYE_COVER_ZONE_TOP,
                    self.EYE_COVER_ZONE_BOTTOM,
                ):
                    if results['left_eye_object'] is None:
                        results['left_eye_object'] = det
                    elif det['conf'] > results['left_eye_object']['conf']:
                        results['left_eye_object'] = det
                if right_eye_allowed and self._bbox_intersects_rect(
                    bbox,
                    self.RIGHT_EYE_X_MIN,
                    self.RIGHT_EYE_X_MAX,
                    self.EYE_COVER_ZONE_TOP,
                    self.EYE_COVER_ZONE_BOTTOM,
                ):
                    if results['right_eye_object'] is None:
                        results['right_eye_object'] = det
                    elif det['conf'] > results['right_eye_object']['conf']:
                        results['right_eye_object'] = det
                continue

            if left_eye_allowed and self.is_left_eye_object(bbox):
                if results['left_eye_object'] is None:
                    results['left_eye_object'] = det
                else:
                    candidate_score = self._compatibility_biased_score(det, 'left_eye')
                    current_score = self._compatibility_biased_score(results['left_eye_object'], 'left_eye')
                    if current_score is None or (
                        candidate_score is not None and (
                            candidate_score > current_score or (
                                candidate_score == current_score and det['conf'] > results['left_eye_object']['conf']
                            )
                        )
                    ):
                        results['left_eye_object'] = det

            if right_eye_allowed and self.is_right_eye_object(bbox):
                if results['right_eye_object'] is None:
                    results['right_eye_object'] = det
                else:
                    candidate_score = self._compatibility_biased_score(det, 'right_eye')
                    current_score = self._compatibility_biased_score(results['right_eye_object'], 'right_eye')
                    if current_score is None or (
                        candidate_score is not None and (
                            candidate_score > current_score or (
                                candidate_score == current_score and det['conf'] > results['right_eye_object']['conf']
                            )
                        )
                    ):
                        results['right_eye_object'] = det

        # 3. Top-face / mouth / shoulder / waist / feet assignments
        for det in non_hand_detections:
            bbox = det['bbox']

            # Needed early for classes 108-109 branch, which now evaluates
            # shoulder/waist/feet before the general branch logic below.
            shoulder_allowed = self._passes_slot_allowlist(det, 'shoulder')
            waist_allowed = self._passes_slot_allowlist(det, 'waist')
            feet_allowed = self._passes_slot_allowlist(det, 'feet')

            if self._is_class_in(det, self.HAND_ONLY_CLASSES):
                # Classes 108-109 are hand-preferred but not hand-exclusive.
                # Allow shoulder, waist, and feet with eligibility guards.
                _h_dists = []
                if left_knuckle != self.DEFAULT_HAND_POSITION:
                    _h_dists.append(self.point_to_bbox_distance(left_knuckle, bbox))
                if right_knuckle != self.DEFAULT_HAND_POSITION:
                    _h_dists.append(self.point_to_bbox_distance(right_knuckle, bbox))
                _near_hand_dist = min(_h_dists) if _h_dists else float('inf')
                _nearhand_penalty = (
                    self.CLASS108109_LOWER_BODY_NEAR_HAND_PENALTY
                    if _near_hand_dist <= self.CLASS108109_NEAR_HAND_DIST
                    else 0.0
                )

                if 'shoulder_object' not in tie_locked_slots and shoulder_allowed and self.CLASS108109_ENABLE_SHOULDER:
                    if self.is_shoulder_object(bbox, left_shoulder, right_shoulder):
                        _score = self._compatibility_biased_score(det, 'shoulder', base_score=det['conf'])
                        if _score is not None:
                            _score += self.CLASS108109_HAND_PREFERENCE_BONUS
                            if results['shoulder_object'] is None:
                                results['shoulder_object'] = det
                            else:
                                _cur = self._compatibility_biased_score(
                                    results['shoulder_object'], 'shoulder',
                                    base_score=results['shoulder_object']['conf']
                                )
                                if _cur is None or _score > _cur or (
                                    _score == _cur and det['conf'] > results['shoulder_object']['conf']
                                ):
                                    results['shoulder_object'] = det

                if waist_allowed and self.CLASS108109_ENABLE_WAIST and self.is_waist_object(bbox):
                    _s_int = self._bbox_intersect_fraction(
                        bbox, self.WAIST_X_MIN, self.WAIST_X_MAX,
                        self.WAIST_ZONE_TOP, self.WAIST_ZONE_BOTTOM,
                    )
                    if _s_int >= self.CLASS108109_WAIST_MIN_INTERSECT:
                        _base = self._compatibility_biased_score(det, 'waist', base_score=det['conf'])
                        if _base is not None:
                            _score = _base + 0.35 * _s_int - _nearhand_penalty
                            if _score >= self.CLASS108109_WAIST_MIN_SCORE:
                                if results['waist_object'] is None:
                                    results['waist_object'] = det
                                else:
                                    _cur_base = self._compatibility_biased_score(
                                        results['waist_object'], 'waist',
                                        base_score=results['waist_object']['conf']
                                    )
                                    _cur_s_int = self._bbox_intersect_fraction(
                                        results['waist_object']['bbox'],
                                        self.WAIST_X_MIN, self.WAIST_X_MAX,
                                        self.WAIST_ZONE_TOP, self.WAIST_ZONE_BOTTOM,
                                    )
                                    _cur_score = (_cur_base or 0.0) + 0.35 * _cur_s_int
                                    if _score > _cur_score or (
                                        _score == _cur_score
                                        and det['conf'] > results['waist_object']['conf']
                                    ):
                                        results['waist_object'] = det

                if feet_allowed and self.CLASS108109_ENABLE_FEET and self.is_feet_object(bbox):
                    _s_int = self._bbox_intersect_fraction(
                        bbox, self.FEET_X_MIN, self.FEET_X_MAX,
                        self.FEET_ZONE_TOP, self.FEET_ZONE_BOTTOM,
                    )
                    if _s_int >= self.CLASS108109_FEET_MIN_INTERSECT:
                        _base = self._compatibility_biased_score(det, 'feet', base_score=det['conf'])
                        if _base is not None:
                            _score = _base + 0.40 * _s_int - _nearhand_penalty
                            if _score >= self.CLASS108109_FEET_MIN_SCORE:
                                if results['feet_object'] is None:
                                    results['feet_object'] = det
                                else:
                                    _cur_base = self._compatibility_biased_score(
                                        results['feet_object'], 'feet',
                                        base_score=results['feet_object']['conf']
                                    )
                                    _cur_s_int = self._bbox_intersect_fraction(
                                        results['feet_object']['bbox'],
                                        self.FEET_X_MIN, self.FEET_X_MAX,
                                        self.FEET_ZONE_TOP, self.FEET_ZONE_BOTTOM,
                                    )
                                    _cur_score = (_cur_base or 0.0) + 0.40 * _cur_s_int
                                    if _score > _cur_score or (
                                        _score == _cur_score
                                        and det['conf'] > results['feet_object']['conf']
                                    ):
                                        results['feet_object'] = det
                continue

            top_face_allowed = self._passes_slot_allowlist(det, 'top_face')
            mouth_allowed = self._passes_slot_allowlist(det, 'mouth')
            shoulder_allowed = self._passes_slot_allowlist(det, 'shoulder')
            waist_allowed = self._passes_slot_allowlist(det, 'waist')
            feet_allowed = self._passes_slot_allowlist(det, 'feet')

            if not top_face_allowed:
                self._record_allowlist_reject('top_face')
            if not mouth_allowed:
                self._record_allowlist_reject('mouth')
            if not shoulder_allowed:
                self._record_allowlist_reject('shoulder')
            if not waist_allowed:
                self._record_allowlist_reject('waist')
            if not feet_allowed:
                self._record_allowlist_reject('feet')

            if self._is_class_in(det, self.COVID_MASK_CLASSES):
                # Intent-aware mask assignment (Rule Spec v1.0).
                # held_in_hand masks skip mouth; worn_on_face and ambiguous evaluate normally.
                _mask_intent = mask_110_intents.get(det['detection_id'], 'ambiguous')
                if _mask_intent != 'held_in_hand' and 'mouth_object' not in tie_locked_slots and mouth_allowed:
                    if self.is_covid_mask_object(bbox):
                        if results['mouth_object'] is None:
                            results['mouth_object'] = det
                        elif det['conf'] > results['mouth_object']['conf']:
                            results['mouth_object'] = det
                continue

            if self._is_class_in(det, self.FULL_FACE_TOP_BIASED_CLASSES):
                if top_face_allowed and (self.is_full_face_mask_object(bbox) or self.is_top_face_object(bbox) or self.is_eye_cover_object(bbox)):
                    if results['top_face_object'] is None:
                        results['top_face_object'] = det
                    elif det['conf'] > results['top_face_object']['conf']:
                        results['top_face_object'] = det
                continue

            if self._is_class_in(det, self.EYE_OR_FOREHEAD_CLASSES):
                if top_face_allowed and self.is_forehead_object(bbox):
                    if results['top_face_object'] is None:
                        results['top_face_object'] = det
                    elif det['conf'] > results['top_face_object']['conf']:
                        results['top_face_object'] = det
                continue

            if self._is_class_in(det, self.EYE_ONLY_CLASSES | self.HAND_OR_EYE_CLASSES):
                continue

            if top_face_allowed and self.is_top_face_object(bbox):
                if results['top_face_object'] is None:
                    results['top_face_object'] = det
                else:
                    candidate_score = self._compatibility_biased_score(det, 'top_face')
                    current_score = self._compatibility_biased_score(results['top_face_object'], 'top_face')
                    if current_score is None or (
                        candidate_score is not None and (
                            candidate_score > current_score or (
                                candidate_score == current_score and bbox['top'] < results['top_face_object']['bbox']['top']
                            )
                        )
                    ):
                        results['top_face_object'] = det

            if 'mouth_object' not in tie_locked_slots and mouth_allowed and self.is_mouth_object(bbox):
                if results['mouth_object'] is None:
                    results['mouth_object'] = det
                else:
                    candidate_score = self._compatibility_biased_score(det, 'mouth')
                    current_score = self._compatibility_biased_score(results['mouth_object'], 'mouth')
                    if current_score is None or (
                        candidate_score is not None and (
                            candidate_score > current_score or (
                                candidate_score == current_score and bbox['top'] < results['mouth_object']['bbox']['top']
                            )
                        )
                    ):
                        results['mouth_object'] = det

            if 'shoulder_object' not in tie_locked_slots and shoulder_allowed and self.is_shoulder_object(bbox, left_shoulder, right_shoulder):
                if results['shoulder_object'] is None:
                    results['shoulder_object'] = det
                else:
                    candidate_score = self._compatibility_biased_score(det, 'shoulder')
                    current_score = self._compatibility_biased_score(results['shoulder_object'], 'shoulder')
                    if current_score is None or (
                        candidate_score is not None and (
                            candidate_score > current_score or (
                                candidate_score == current_score and det['conf'] > results['shoulder_object']['conf']
                            )
                        )
                    ):
                        results['shoulder_object'] = det

            if waist_allowed and self.is_waist_object(bbox):
                candidate_score, reject_reason = lower_body_candidate_score(det, 'waist')
                if candidate_score is None:
                    self._record_unassigned_reason('waist', reject_reason)
                elif results['waist_object'] is None:
                    results['waist_object'] = det
                else:
                    current_score, _ = lower_body_candidate_score(results['waist_object'], 'waist')
                    if current_score is None or candidate_score > current_score or (
                        candidate_score == current_score and det['conf'] > results['waist_object']['conf']
                    ):
                        results['waist_object'] = det

            if feet_allowed and self.is_feet_object(bbox):
                candidate_score, reject_reason = lower_body_candidate_score(det, 'feet')
                if candidate_score is None:
                    self._record_unassigned_reason('feet', reject_reason)
                elif results['feet_object'] is None:
                    results['feet_object'] = det
                else:
                    current_score, _ = lower_body_candidate_score(results['feet_object'], 'feet')
                    if current_score is None or candidate_score > current_score or (
                        candidate_score == current_score and det['conf'] > results['feet_object']['conf']
                    ):
                        results['feet_object'] = det

        assigned_ids_by_class = {class_id: set() for class_id in self.DEBUG_TRACKED_CLASS_IDS}
        for slot_det in results.values():
            if slot_det is None:
                continue
            class_id = self._get_detection_class_id(slot_det)
            if class_id in assigned_ids_by_class:
                assigned_ids_by_class[class_id].add(int(slot_det['detection_id']))

        for class_id in self.DEBUG_TRACKED_CLASS_IDS:
            self._class_pipeline_debug_counts[class_id]['final_assigned_unique'] += len(assigned_ids_by_class[class_id])
        
        return results

    def query_and_classify_detections(
        self,
        image_id,
        left_knuckle,
        right_knuckle,
        left_shoulder=None,
        right_shoulder=None,
        body_landmarks_normalized=None,
    ):
        """
        Query detections for an image and classify their relationship to hands/face.
        Returns dict with 9 keys, each containing a detection payload dict or None.
        """
        if self.session is None:
            raise ValueError("Session not initialized. Pass session to ToolsClustering.__init__()")
        
        # Query detections
        detection_results = self.session.query(Detections).filter_by(image_id=image_id).\
            filter(Detections.conf > self.MIN_DETECTION_CONFIDENCE).all()
        
        debug = self.VERBOSE  # print full math when VERBOSE is on
        if debug:
            print(f"\n{'='*60}")
            print(f"[DEBUG] query_and_classify_detections image_id={image_id}")
            # print(f"  left_knuckle={left_knuckle}  right_knuckle={right_knuckle}")
            # print(f"  left_shoulder={left_shoulder}  right_shoulder={right_shoulder}")
            # print(f"  Raw detection_results count: {len(detection_results)}")
            # for d in detection_results:
            #     print(f"    detection_id={d.detection_id} class_id={d.class_id} conf={d.conf:.3f} bbox_norm={d.bbox_norm}")

        if not detection_results:
            if debug: print(f"  → No detections above MIN_DETECTION_CONFIDENCE={self.MIN_DETECTION_CONFIDENCE}; returning all None")
            return {
                'left_hand_object': None,
                'right_hand_object': None,
                'top_face_object': None,
                'left_eye_object': None,
                'right_eye_object': None,
                'mouth_object': None,
                'shoulder_object': None,
                'waist_object': None,
                'feet_object': None,
            }
        
        # Parse detections into standardized format
        detections = []
        for d in detection_results:
            class_id_value = self._normalize_class_id(getattr(d, 'class_id', None))
            if class_id_value in self._class_pipeline_debug_counts:
                self._class_pipeline_debug_counts[class_id_value]['query_hits'] += 1

            bbox = self.parse_bbox_norm(d.bbox_norm)
            if bbox is None:
                if class_id_value in self._class_pipeline_debug_counts:
                    self._class_pipeline_debug_counts[class_id_value]['bbox_parse_fail'] += 1
                    examples = self._class_pipeline_debug_counts[class_id_value]['parse_fail_examples']
                    if len(examples) < 3:
                        raw_preview = str(d.bbox_norm)
                        examples.append(raw_preview[:160])
                # if debug: print(f"  ✗ detection_id={d.detection_id} — bbox_norm parse failed, skipping")
                continue

            if class_id_value in self._class_pipeline_debug_counts:
                self._class_pipeline_debug_counts[class_id_value]['bbox_parse_ok'] += 1

            detections.append({
                'detection_id': d.detection_id,
                'class_id': d.class_id,
                'class_id_raw': d.class_id,
                'conf': d.conf,
                'bbox': bbox,
                'top': bbox['top'],
                'left': bbox['left'],
                'right': bbox['right'],
                'bottom': bbox['bottom']
            })
            if debug:
                print(f"  ✓ detection_id={d.detection_id} parsed: class_id={d.class_id} conf={d.conf:.3f}")
                print(f"    bbox → top={bbox['top']} left={bbox['left']} right={bbox['right']} bottom={bbox['bottom']}")
                # Check each classifier
                print(f"    is_top_face_object:    {self.is_top_face_object(bbox)}")
                print(f"      (top<0={bbox['top']<0}, left<0={bbox['left']<0}, right>0={bbox['right']>0}, width={bbox['right']-bbox['left']:.4f}<=MAX={self.MAX_FACE_WIDTH}, extends_into_bottom={max(0,bbox['bottom']):.4f}<=MAX_VERT={self.MAX_FACE_VERT_EXTENSION})")
                print(f"    is_bottom_face_object: {self.is_bottom_face_object(bbox)}")
                print(f"    is_left_eye_object:    {self.is_left_eye_object(bbox)}")
                print(f"    is_right_eye_object:   {self.is_right_eye_object(bbox)}")
                print(f"    is_full_face_mask:     {self.is_full_face_mask_object(bbox)}")
                print(f"    is_mouth_object:       {self.is_mouth_object(bbox)}")
                print(f"    is_shoulder_object:    {self.is_shoulder_object(bbox, left_shoulder, right_shoulder)}")
                print(f"    is_waist_object:       {self.is_waist_object(bbox)}")
                print(f"    is_feet_object:        {self.is_feet_object(bbox)}")
                dist_left  = self.point_to_bbox_distance(left_knuckle, bbox)  if left_knuckle  != self.DEFAULT_HAND_POSITION else None
                dist_right = self.point_to_bbox_distance(right_knuckle, bbox) if right_knuckle != self.DEFAULT_HAND_POSITION else None
                print(f"    dist left_knuckle→bbox:  {dist_left}  (TOUCH_THRESHOLD={self.TOUCH_THRESHOLD})")
                print(f"    dist right_knuckle→bbox: {dist_right}")

        if debug:
            print(f"  Parsed {len(detections)} valid detections (of {len(detection_results)} raw)")

        # Classify relationships using class method
        classified = self.classify_object_hand_relationships(
            detections,
            left_knuckle,
            right_knuckle,
            left_shoulder=left_shoulder,
            right_shoulder=right_shoulder,
            body_landmarks_normalized=body_landmarks_normalized,
        )

        # Track where classes 110-119 are lost: seen in detections vs assigned to any slot.
        seen_ids_by_class = {class_id: set() for class_id in self.DEBUG_TRACKED_CLASS_IDS}
        assigned_ids_by_class = {class_id: set() for class_id in self.DEBUG_TRACKED_CLASS_IDS}

        for det in detections:
            class_id = self._get_detection_class_id(det)
            if class_id in seen_ids_by_class:
                seen_ids_by_class[class_id].add(int(det['detection_id']))

        for slot_name, assigned_det in classified.items():
            if assigned_det is None:
                continue
            class_id = self._get_detection_class_id(assigned_det)
            if class_id not in self._class_assignment_debug_counts:
                continue

            self._class_assignment_debug_counts[class_id]['assigned_slot_hits'] += 1
            if slot_name in self._class_assignment_debug_counts[class_id]['slot_hits']:
                self._class_assignment_debug_counts[class_id]['slot_hits'][slot_name] += 1
            assigned_ids_by_class[class_id].add(int(assigned_det['detection_id']))

        for class_id in self.DEBUG_TRACKED_CLASS_IDS:
            seen_count = len(seen_ids_by_class[class_id])
            assigned_unique_count = len(assigned_ids_by_class[class_id])
            dropped_count = max(0, seen_count - assigned_unique_count)

            self._class_assignment_debug_counts[class_id]['seen'] += seen_count
            self._class_assignment_debug_counts[class_id]['assigned_unique'] += assigned_unique_count
            self._class_assignment_debug_counts[class_id]['dropped'] += dropped_count

        if debug:
            print(f"  Classification results:")
            for k, v in classified.items():
                print(f"    {k}: {v}")
        
        # Convert to compact payload dicts for df storage / DB persistence
        result = {}
        for key, det in classified.items():
            if det is None:
                result[key] = None
            else:
                result[key] = self.detection_to_payload(det)
        
        return result

    def process_detections_for_df(self, df):
        """
        Process all detections for a dataframe and add object classification columns.
        Expects df to have: image_id, left_pointer_knuckle_norm, right_pointer_knuckle_norm
        """
        self.reset_allowlist_reject_counts()
        self.reset_unassigned_reason_counts()
        self.reset_class_assignment_debug_counts()
        self.reset_class_pipeline_debug_counts()

        # Initialize new columns
        df['left_hand_object'] = None
        df['right_hand_object'] = None
        df['top_face_object'] = None
        df['left_eye_object'] = None
        df['right_eye_object'] = None
        df['mouth_object'] = None
        df['shoulder_object'] = None
        df['waist_object'] = None
        df['feet_object'] = None
        
        for idx, row in df.iterrows():
            image_id = row['image_id']
            left_knuckle = row.get('left_pointer_knuckle_norm', self.DEFAULT_HAND_POSITION)
            right_knuckle = row.get('right_pointer_knuckle_norm', self.DEFAULT_HAND_POSITION)
            
            # Handle case where knuckle data might be None or string
            if left_knuckle is None or (isinstance(left_knuckle, list) and len(left_knuckle) == 0):
                left_knuckle = self.DEFAULT_HAND_POSITION
            if right_knuckle is None or (isinstance(right_knuckle, list) and len(right_knuckle) == 0):
                right_knuckle = self.DEFAULT_HAND_POSITION

            left_shoulder, right_shoulder = self.extract_shoulder_points(row.get('body_landmarks_normalized'))
            
            # Query and classify
            classifications = self.query_and_classify_detections(
                image_id,
                left_knuckle,
                right_knuckle,
                left_shoulder=left_shoulder,
                right_shoulder=right_shoulder,
                body_landmarks_normalized=row.get('body_landmarks_normalized'),
            )
            
            # Assign to df
            df.at[idx, 'left_hand_object'] = classifications['left_hand_object']
            df.at[idx, 'right_hand_object'] = classifications['right_hand_object']
            df.at[idx, 'top_face_object'] = classifications['top_face_object']
            df.at[idx, 'left_eye_object'] = classifications['left_eye_object']
            df.at[idx, 'right_eye_object'] = classifications['right_eye_object']
            df.at[idx, 'mouth_object'] = classifications['mouth_object']
            df.at[idx, 'shoulder_object'] = classifications['shoulder_object']
            df.at[idx, 'waist_object'] = classifications['waist_object']
            df.at[idx, 'feet_object'] = classifications['feet_object']
        
        return df

    def process_detections_with_debug_reporting(self, df, engine, label="", tracked_class_ids=None, conf_threshold=None):
        """
        Run ObjectFusion detection assignment with consolidated debug/report logging.
        Returns processed dataframe with object assignment columns filled.
        """
        tracked_ids = tuple(tracked_class_ids) if tracked_class_ids is not None else self.DEBUG_TRACKED_CLASS_IDS
        object_assignment_cols = [
            'left_hand_object', 'right_hand_object', 'top_face_object',
            'left_eye_object', 'right_eye_object', 'mouth_object', 'shoulder_object',
            'waist_object', 'feet_object'
        ]

        # Pre-assignment scope counts from Detections table.
        image_ids = []
        if 'image_id' in df.columns:
            image_ids = df['image_id'].dropna().astype(int).tolist()

        tracked_scope_counts = self.count_tracked_detections_for_image_ids(
            engine,
            image_ids,
            class_ids=tracked_ids,
            conf_threshold=conf_threshold,
        )
        tracked_scope_total_rows = int(sum(v['det_rows'] for v in tracked_scope_counts.values()))
        tracked_scope_total_images = int(sum(v['image_rows'] for v in tracked_scope_counts.values()))
        print(
            f"{label}[COUNT] Tracked classes present in Detections for this df scope "
            f"(class 110-119, conf>={conf_threshold}, bbox_norm not null): "
            f"det_rows={tracked_scope_total_rows}, image_rows_sum={tracked_scope_total_images}"
        )
        for class_id in tracked_ids:
            class_counts = tracked_scope_counts.get(class_id, {'det_rows': 0, 'image_rows': 0})
            print(
                f"{label}[COUNT] Detections scope class {class_id}: "
                f"det_rows={int(class_counts['det_rows'])}, image_rows={int(class_counts['image_rows'])}"
            )

        # Assignment pass.
        df = self.process_detections_for_df(df)

        tracked_counts = self.get_class_assignment_debug_counts()
        print(f"{label}[COUNT] Class 110-119 assignment summary (seen/assigned_unique/dropped/slot_hits):")
        for class_id in tracked_ids:
            stats = tracked_counts.get(class_id, {})
            seen_count = int(stats.get('seen', 0))
            assigned_unique_count = int(stats.get('assigned_unique', 0))
            dropped_count = int(stats.get('dropped', 0))
            slot_hits_total = int(stats.get('assigned_slot_hits', 0))
            slot_hits = stats.get('slot_hits', {}) if isinstance(stats.get('slot_hits', {}), dict) else {}
            compact_slot_hits = {slot: int(count) for slot, count in slot_hits.items() if int(count) > 0}
            print(
                f"{label}[COUNT] class {class_id}: "
                f"seen={seen_count}, assigned_unique={assigned_unique_count}, "
                f"dropped={dropped_count}, slot_hits={slot_hits_total}, by_slot={compact_slot_hits if compact_slot_hits else '{}'}"
            )

        pipeline_counts = self.get_class_pipeline_debug_counts()
        print(f"{label}[COUNT] Class 110-119 pipeline stages (query -> parse -> resolve -> tie/non-hand -> final):")
        for class_id in tracked_ids:
            stats = pipeline_counts.get(class_id, {})
            print(
                f"{label}[COUNT] class {class_id}: "
                f"query_hits={int(stats.get('query_hits', 0))}, "
                f"bbox_parse_ok={int(stats.get('bbox_parse_ok', 0))}, "
                f"bbox_parse_fail={int(stats.get('bbox_parse_fail', 0))}, "
                f"post_resolve={int(stats.get('post_resolve', 0))}, "
                f"tie_blocked={int(stats.get('tie_blocked', 0))}, "
                f"non_hand_pool={int(stats.get('non_hand_pool', 0))}, "
                f"final_assigned_unique={int(stats.get('final_assigned_unique', 0))}"
            )
            fail_examples = stats.get('parse_fail_examples', []) if isinstance(stats.get('parse_fail_examples', []), list) else []
            if fail_examples:
                print(f"{label}[COUNT] class {class_id} parse_fail_examples: {fail_examples}")

        if self.USE_ALLOWLIST:
            allowlist_rejects = self.get_allowlist_reject_counts()
            compact_rejects = {k: int(v) for k, v in allowlist_rejects.items() if int(v) > 0}
            print(f"{label}[COUNT] Allowlist rejects by slot: {compact_rejects if compact_rejects else 'none'}")

        unassigned_reasons = self.get_unassigned_reason_counts()
        compact_unassigned = {
            slot: reasons
            for slot, reasons in unassigned_reasons.items()
            if reasons
        }
        print(
            f"{label}[COUNT] Explicit unassigned lower-body reasons: "
            f"{compact_unassigned if compact_unassigned else 'none'}"
        )

        rows_with_any_object = int(df[object_assignment_cols].notna().any(axis=1).sum())
        rows_with_no_objects = int(len(df) - rows_with_any_object)
        print(f"{label}[COUNT] Rows with any object assignment: {rows_with_any_object}")
        print(f"{label}[COUNT] Rows with no object assignments: {rows_with_no_objects}")
        for col in object_assignment_cols:
            print(f"{label}[COUNT] {col} assigned: {int(df[col].notna().sum())}")

        return df

    def _build_detection_payload_from_sql_fields(self, detection_id, class_id, conf, bbox_norm):
        """Build detection payload dict from SQL selected fields."""
        if detection_id is None or class_id is None or conf is None or bbox_norm is None:
            return None

        bbox = self.parse_bbox_norm(bbox_norm)
        if bbox is None:
            return None

        required_keys = ['top', 'left', 'right', 'bottom']
        if not all(k in bbox for k in required_keys):
            return None

        return {
            'detection_id': int(detection_id),
            'class_id': float(class_id),
            'conf': float(conf),
            'top': float(bbox['top']),
            'left': float(bbox['left']),
            'right': float(bbox['right']),
            'bottom': float(bbox['bottom']),
        }

    def get_precomputed_detections_by_image_ids(self, image_ids):
        """
        Read precomputed detection assignments from ImagesDetections and join to Detections.
        Returns dict keyed by image_id with values containing the 9 detection payload fields.
        """
        if self.session is None:
            raise ValueError("Session not initialized. Pass session to ToolsClustering.__init__()")

        if not image_ids:
            return {}

        sql = text("""
            SELECT
                idet.image_id,

                lh.detection_id AS left_hand_detection_id,
                lh.class_id AS left_hand_class_id,
                lh.conf AS left_hand_conf,
                lh.bbox_norm AS left_hand_bbox_norm,

                rh.detection_id AS right_hand_detection_id,
                rh.class_id AS right_hand_class_id,
                rh.conf AS right_hand_conf,
                rh.bbox_norm AS right_hand_bbox_norm,

                tf.detection_id AS top_face_detection_id,
                tf.class_id AS top_face_class_id,
                tf.conf AS top_face_conf,
                tf.bbox_norm AS top_face_bbox_norm,

                le.detection_id AS left_eye_detection_id,
                le.class_id AS left_eye_class_id,
                le.conf AS left_eye_conf,
                le.bbox_norm AS left_eye_bbox_norm,

                re.detection_id AS right_eye_detection_id,
                re.class_id AS right_eye_class_id,
                re.conf AS right_eye_conf,
                re.bbox_norm AS right_eye_bbox_norm,

                mo.detection_id AS mouth_detection_id,
                mo.class_id AS mouth_class_id,
                mo.conf AS mouth_conf,
                mo.bbox_norm AS mouth_bbox_norm,

                so.detection_id AS shoulder_detection_id,
                so.class_id AS shoulder_class_id,
                so.conf AS shoulder_conf,
                so.bbox_norm AS shoulder_bbox_norm,

                wa.detection_id AS waist_detection_id,
                wa.class_id AS waist_class_id,
                wa.conf AS waist_conf,
                wa.bbox_norm AS waist_bbox_norm,

                fe.detection_id AS feet_detection_id,
                fe.class_id AS feet_class_id,
                fe.conf AS feet_conf,
                fe.bbox_norm AS feet_bbox_norm
            FROM ImagesDetections idet
            LEFT JOIN Detections lh ON idet.left_hand_object_id = lh.detection_id
            LEFT JOIN Detections rh ON idet.right_hand_object_id = rh.detection_id
            LEFT JOIN Detections tf ON idet.top_face_object_id = tf.detection_id
            LEFT JOIN Detections le ON idet.left_eye_object_id = le.detection_id
            LEFT JOIN Detections re ON idet.right_eye_object_id = re.detection_id
            LEFT JOIN Detections mo ON idet.mouth_object_id = mo.detection_id
            LEFT JOIN Detections so ON idet.shoulder_object_id = so.detection_id
            LEFT JOIN Detections wa ON idet.waist_object_id = wa.detection_id
            LEFT JOIN Detections fe ON idet.feet_object_id = fe.detection_id
            WHERE idet.image_id IN :image_ids
        """).bindparams(bindparam("image_ids", expanding=True))

        rows = self.session.execute(sql, {"image_ids": list(image_ids)}).mappings().all()

        result = {}
        for row in rows:
            image_id = row['image_id']
            result[image_id] = {
                'left_hand_object': self._build_detection_payload_from_sql_fields(
                    row['left_hand_detection_id'], row['left_hand_class_id'], row['left_hand_conf'], row['left_hand_bbox_norm']
                ),
                'right_hand_object': self._build_detection_payload_from_sql_fields(
                    row['right_hand_detection_id'], row['right_hand_class_id'], row['right_hand_conf'], row['right_hand_bbox_norm']
                ),
                'top_face_object': self._build_detection_payload_from_sql_fields(
                    row['top_face_detection_id'], row['top_face_class_id'], row['top_face_conf'], row['top_face_bbox_norm']
                ),
                'left_eye_object': self._build_detection_payload_from_sql_fields(
                    row['left_eye_detection_id'], row['left_eye_class_id'], row['left_eye_conf'], row['left_eye_bbox_norm']
                ),
                'right_eye_object': self._build_detection_payload_from_sql_fields(
                    row['right_eye_detection_id'], row['right_eye_class_id'], row['right_eye_conf'], row['right_eye_bbox_norm']
                ),
                'mouth_object': self._build_detection_payload_from_sql_fields(
                    row['mouth_detection_id'], row['mouth_class_id'], row['mouth_conf'], row['mouth_bbox_norm']
                ),
                'shoulder_object': self._build_detection_payload_from_sql_fields(
                    row['shoulder_detection_id'], row['shoulder_class_id'], row['shoulder_conf'], row['shoulder_bbox_norm']
                ),
                'waist_object': self._build_detection_payload_from_sql_fields(
                    row['waist_detection_id'], row['waist_class_id'], row['waist_conf'], row['waist_bbox_norm']
                ),
                'feet_object': self._build_detection_payload_from_sql_fields(
                    row['feet_detection_id'], row['feet_class_id'], row['feet_conf'], row['feet_bbox_norm']
                ),
            }

        return result

    def ensure_images_armsposes3d_table(self):
        """Ensure cache table for subsetted arms pose vectors exists."""
        if self.session is None:
            raise ValueError("Session not initialized. Pass session to ToolsClustering.__init__()")

        create_sql = text(f"""
            CREATE TABLE IF NOT EXISTS {self.ARMS_POSE_CACHE_TABLE} (
                image_id BIGINT NOT NULL,
                arms_subset_json JSON NULL,
                has_world_lms TINYINT(1) NOT NULL DEFAULT 0,
                subset_name VARCHAR(64) NOT NULL,
                subset_version INT NOT NULL,
                updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                PRIMARY KEY (image_id)
            )
        """)
        self.session.execute(create_sql)

        # Backfill legacy table schemas in-place when table already exists without
        # the cache columns needed by ObjectFusion hydration/upsert.
        existing_cols_sql = text(f"SHOW COLUMNS FROM {self.ARMS_POSE_CACHE_TABLE}")
        existing_cols_rows = self.session.execute(existing_cols_sql).fetchall()
        existing_cols = {row[0] for row in existing_cols_rows}

        if 'arms_subset_json' not in existing_cols:
            self.session.execute(text(
                f"ALTER TABLE {self.ARMS_POSE_CACHE_TABLE} ADD COLUMN arms_subset_json JSON NULL"
            ))
        if 'has_world_lms' not in existing_cols:
            self.session.execute(text(
                f"ALTER TABLE {self.ARMS_POSE_CACHE_TABLE} ADD COLUMN has_world_lms TINYINT(1) NOT NULL DEFAULT 0"
            ))
        if 'subset_name' not in existing_cols:
            self.session.execute(text(
                f"ALTER TABLE {self.ARMS_POSE_CACHE_TABLE} ADD COLUMN subset_name VARCHAR(64) NOT NULL DEFAULT '{self.ARMS_POSE_SUBSET_NAME}'"
            ))
        if 'subset_version' not in existing_cols:
            self.session.execute(text(
                f"ALTER TABLE {self.ARMS_POSE_CACHE_TABLE} ADD COLUMN subset_version INT NOT NULL DEFAULT {int(self.ARMS_POSE_SUBSET_VERSION)}"
            ))
        if 'updated_at' not in existing_cols:
            self.session.execute(text(
                f"ALTER TABLE {self.ARMS_POSE_CACHE_TABLE} ADD COLUMN updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP"
            ))

    def get_precomputed_armsposes_by_image_ids(self, image_ids):
        """
        Read precomputed subsetted arms pose vectors from ImagesArmsFeatures3D.
        Returns dict keyed by image_id with cached payload and flags.
        """
        if self.session is None:
            raise ValueError("Session not initialized. Pass session to ToolsClustering.__init__()")

        if not image_ids:
            return {}

        self.ensure_images_armsposes3d_table()

        sql = text(f"""
            SELECT
                image_id,
                arms_subset_json,
                has_world_lms,
                subset_name,
                subset_version
            FROM {self.ARMS_POSE_CACHE_TABLE}
            WHERE image_id IN :image_ids
        """).bindparams(bindparam("image_ids", expanding=True))

        rows = self.session.execute(sql, {"image_ids": list(image_ids)}).mappings().all()

        result = {}
        for row in rows:
            raw_payload = row.get('arms_subset_json')
            payload = None
            if raw_payload is not None:
                if isinstance(raw_payload, str):
                    try:
                        payload = json.loads(raw_payload)
                    except json.JSONDecodeError:
                        payload = None
                else:
                    payload = raw_payload

            image_id = int(row['image_id'])
            result[image_id] = {
                'arms_subset_vector': payload,
                'has_world_lms': bool(row.get('has_world_lms', 0)),
                'subset_name': row.get('subset_name'),
                'subset_version': row.get('subset_version'),
            }

        return result

    def hydrate_detections_from_precomputed_table(self, df):
        """
        Fill detection classification columns from ImagesDetections + Detections.
        Returns (updated_df, missing_image_ids) where missing_image_ids are not found
        in ImagesDetections at all.
        """
        required_cols = [
            'left_hand_object', 'right_hand_object',
            'top_face_object', 'left_eye_object', 'right_eye_object',
            'mouth_object', 'shoulder_object', 'waist_object', 'feet_object'
        ]
        for col in required_cols:
            if col not in df.columns:
                df[col] = None

        image_ids = [int(x) for x in df['image_id'].dropna().unique().tolist()]
        precomputed = self.get_precomputed_detections_by_image_ids(image_ids)

        for idx, row in df.iterrows():
            image_id = row.get('image_id')
            if image_id is None:
                continue
            payload = precomputed.get(int(image_id))
            if payload is None:
                continue

            df.at[idx, 'left_hand_object'] = payload['left_hand_object']
            df.at[idx, 'right_hand_object'] = payload['right_hand_object']
            df.at[idx, 'top_face_object'] = payload['top_face_object']
            df.at[idx, 'left_eye_object'] = payload['left_eye_object']
            df.at[idx, 'right_eye_object'] = payload['right_eye_object']
            df.at[idx, 'mouth_object'] = payload['mouth_object']
            df.at[idx, 'shoulder_object'] = payload['shoulder_object']
            df.at[idx, 'waist_object'] = payload['waist_object']
            df.at[idx, 'feet_object'] = payload['feet_object']

        missing_image_ids = sorted(list(set(image_ids) - set(precomputed.keys())))
        return df, missing_image_ids

    def hydrate_armsposes_from_precomputed_table(self, df):
        """
        Fill subsetted arms pose columns from ImagesArmsFeatures3D.
        Returns (updated_df, missing_image_ids) where missing image IDs are not found in cache.
        """
        if 'arms_subset_vector' not in df.columns:
            df['arms_subset_vector'] = None
        if 'has_world_lms' not in df.columns:
            df['has_world_lms'] = False

        image_ids = [int(x) for x in df['image_id'].dropna().unique().tolist()]
        precomputed = self.get_precomputed_armsposes_by_image_ids(image_ids)

        for idx, row in df.iterrows():
            image_id = row.get('image_id')
            if image_id is None:
                continue
            payload = precomputed.get(int(image_id))
            if payload is None:
                continue

            df.at[idx, 'arms_subset_vector'] = payload.get('arms_subset_vector')
            df.at[idx, 'has_world_lms'] = bool(payload.get('has_world_lms', False))

        missing_image_ids = sorted(list(set(image_ids) - set(precomputed.keys())))
        return df, missing_image_ids

    def convert_arms_subset_vector_to_dim_columns(self, df, source_col='arms_subset_vector', drop_source=False):
        """
        Convert flattened arms subset vectors into dim_* columns used by clustering.
        """
        if df is None or len(df) == 0:
            return df
        if source_col not in df.columns:
            return df

        def _as_sequence(value):
            if isinstance(value, np.ndarray):
                if value.size == 0:
                    return None
                return value.flatten().tolist()
            if isinstance(value, (list, tuple)):
                if len(value) == 0:
                    return None
                return list(value)
            return None

        dim_count = 0
        for value in df[source_col]:
            seq = _as_sequence(value)
            if seq is not None:
                dim_count = len(seq)
                break

        if dim_count == 0:
            return df

        def _dim_value(value, dim_idx):
            seq = _as_sequence(value)
            if seq is None or len(seq) <= dim_idx:
                return 0.0
            item = seq[dim_idx]
            return float(item) if item is not None else 0.0

        for dim_idx in range(dim_count):
            col_name = f"dim_{dim_idx}"
            df[col_name] = df[source_col].apply(lambda value: _dim_value(value, dim_idx))

        if drop_source:
            df = df.drop(columns=[source_col], errors='ignore')

        return df

    def persist_images_armsposes3d(self, df):
        """Upsert subsetted arms pose vectors into ImagesArmsFeatures3D cache table."""
        if self.session is None:
            raise ValueError("Session not initialized. Pass session to ToolsClustering.__init__()")

        if df is None or len(df) == 0:
            return 0

        self.ensure_images_armsposes3d_table()

        upsert_sql = text(f"""
            INSERT INTO {self.ARMS_POSE_CACHE_TABLE}
                (image_id, arms_subset_json, has_world_lms, subset_name, subset_version, updated_at)
            VALUES
                (:image_id, :arms_subset_json, :has_world_lms, :subset_name, :subset_version, CURRENT_TIMESTAMP)
            ON DUPLICATE KEY UPDATE
                arms_subset_json = VALUES(arms_subset_json),
                has_world_lms = VALUES(has_world_lms),
                subset_name = VALUES(subset_name),
                subset_version = VALUES(subset_version),
                updated_at = CURRENT_TIMESTAMP
        """)

        rows_written = 0
        for _, row in df.iterrows():
            image_id = row.get('image_id')
            if image_id is None or (isinstance(image_id, float) and np.isnan(image_id)):
                continue

            subset_vector = row.get('arms_subset_vector')
            subset_json = None
            if subset_vector is not None:
                try:
                    subset_json = json.dumps(list(subset_vector))
                except TypeError:
                    subset_json = None

            has_world_lms = row.get('has_world_lms')
            has_world_lms = bool(has_world_lms) if has_world_lms is not None else False

            self.session.execute(upsert_sql, {
                'image_id': int(image_id),
                'arms_subset_json': subset_json,
                'has_world_lms': 1 if has_world_lms else 0,
                'subset_name': self.ARMS_POSE_SUBSET_NAME,
                'subset_version': int(self.ARMS_POSE_SUBSET_VERSION),
            })
            rows_written += 1

        return rows_written

    def prepare_objectfusion_pose_features(self, df, sort, io, structure="list3D", label=""):
        """
        Prepare ObjectFusion pose features from Mongo payloads:
        - hand landmark unpack
        - world-landmark subset vector build
        - world-lms validity filtering
        - knuckle extraction + source diagnostics
        """
        if df is None or len(df) == 0:
            return None

        # Replace NaN values with None before applying prep functions.
        df['hand_results'] = df['hand_results'].apply(lambda x: None if pd.isna(x) else x)
        df[[
            'left_hand_landmarks', 'left_hand_world_landmarks', 'left_hand_landmarks_norm',
            'right_hand_landmarks', 'right_hand_world_landmarks', 'right_hand_landmarks_norm'
        ]] = pd.DataFrame(df['hand_results'].apply(sort.prep_hand_landmarks).tolist(), index=df.index)

        if len(df) > 0 and len(df['left_hand_landmarks_norm'].iloc[0]) > 0:
            print(f"Sample left_hand_landmarks_norm (first landmark): {df['left_hand_landmarks_norm'].iloc[0][0]}")

        print("\n[DEBUG] Extracting finger positions from body landmarks...")
        print(f"[DEBUG] df shape: {df.shape}")

        print("[DEBUG] Unpickling body_landmarks_normalized...")
        df['body_landmarks_normalized'] = df['body_landmarks_normalized'].apply(io.unpickle_array)
        if not self.SUPPRESS_ARMS_FEATURES:
            print("[DEBUG] Unpickling body_landmarks_3D and building subsetted arms vectors...")
            df['body_landmarks_3D'] = df['body_landmarks_3D'].apply(io.unpickle_array)

            arms_subset_indices = sort.make_subset_landmarks(0, 22)
            max_subset_index = max(arms_subset_indices)
            df['body_landmarks_array_3d'] = df['body_landmarks_3D'].apply(
                lambda x: sort.get_landmarks_2d(x, list(range(33)), structure=structure)
            )
            df['arms_subset_vector'] = df['body_landmarks_array_3d'].apply(
                lambda arr: [arr[i] for i in arms_subset_indices]
                if isinstance(arr, (list, tuple, np.ndarray)) and len(arr) > max_subset_index
                else None
            )
            df['has_world_lms'] = df['arms_subset_vector'].notna()

            invalid_subset_df = df[df['has_world_lms'] == False]
            invalid_subset_count = int(len(invalid_subset_df))
            self.increment_world_lms_stat('excluded_invalid_subset', invalid_subset_count)
            if invalid_subset_count > 0 and 'image_id' in invalid_subset_df.columns:
                for image_id in invalid_subset_df['image_id'].head(self.WORLD_LMS_REPORT_MAX_SAMPLES):
                    self.append_world_lms_sample_id('excluded_invalid_subset_sample_ids', image_id)
                print(f"{label}[COUNT] Rows excluded due to invalid arms subset vectors: {invalid_subset_count}")

            df = df[df['has_world_lms'] == True].copy()
            if len(df) == 0:
                print("WARNING: No rows with valid arms subset vectors found. Cannot cluster.")
                return None

        knuckle_results = df.apply(
            lambda row: sort.prep_knuckle_landmarks(row['hand_results'], row['body_landmarks_normalized']),
            axis=1
        ).tolist()
        print(f"[DEBUG] prep_knuckle_landmarks returned {len(knuckle_results)} results")
        if len(knuckle_results) > 0:
            print(f"[DEBUG] Sample result (first row): {knuckle_results[0]}")

        df[["left_pointer_knuckle_norm", "right_pointer_knuckle_norm", "left_source", "right_source"]] = pd.DataFrame(
            knuckle_results,
            index=df.index
        )

        left_body_count = int((df['left_source'] == 'body').sum())
        right_body_count = int((df['right_source'] == 'body').sum())
        left_default_count = int((df['left_source'] == 'default').sum())
        right_default_count = int((df['right_source'] == 'default').sum())
        left_nondefault_count = int(df['left_pointer_knuckle_norm'].apply(
            lambda value: value != [0.0, 8.0, 0.0] if isinstance(value, list) else False
        ).sum())
        right_nondefault_count = int(df['right_pointer_knuckle_norm'].apply(
            lambda value: value != [0.0, 8.0, 0.0] if isinstance(value, list) else False
        ).sum())
        print(f"{label}[COUNT] Left knuckles from body/default: {left_body_count}/{left_default_count}")
        print(f"{label}[COUNT] Right knuckles from body/default: {right_body_count}/{right_default_count}")
        print(f"{label}[COUNT] Rows with non-default left/right knuckles: {left_nondefault_count}/{right_nondefault_count}")

        left_source_counts = df['left_source'].value_counts()
        right_source_counts = df['right_source'].value_counts()
        print(f"[DEBUG] Left finger position sources: {left_source_counts.to_dict()}")
        print(f"[DEBUG] Right finger position sources: {right_source_counts.to_dict()}")
        print(f"[DEBUG] Sample left_pointer_knuckle_norm (first row): {df['left_pointer_knuckle_norm'].iloc[0] if len(df) > 0 else 'N/A'}")
        print(f"[DEBUG] Sample right_pointer_knuckle_norm (first row): {df['right_pointer_knuckle_norm'].iloc[0] if len(df) > 0 else 'N/A'}")

        return df

    def flatten_object_detections(self, detection_payload):
        """
        Flatten a detection payload dict into features.
        Returns a 6-element list, or [0,0,0,0,0,0] if None.
        """
        if detection_payload is None:
            return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        if isinstance(detection_payload, dict):
            return [
                float(detection_payload.get('class_id', 0.0)),
                float(detection_payload.get('conf', 0.0)),
                float(detection_payload.get('top', 0.0)),
                float(detection_payload.get('left', 0.0)),
                float(detection_payload.get('right', 0.0)),
                float(detection_payload.get('bottom', 0.0)),
            ]
        return list(detection_payload)

    def prepare_features_for_knn(self, df):
        """
        Flatten all columns into a single feature vector for KNN clustering.
        Handles numeric columns and object detection lists.
        Returns df with all features flattened into separate columns (columnar format).
        """
        import pandas as pd
        
        # Face angles are retained upstream but excluded from ObjectFusion clustering features.
        numeric_cols = []
        
        # Detection columns (6 values each: class_id, conf, top, left, right, bottom)
        detection_cols = ['left_hand_object', 'right_hand_object', 'top_face_object',
                  'left_eye_object', 'right_eye_object', 'mouth_object', 'shoulder_object',
                  'waist_object', 'feet_object']
        detection_fields = ['class_id', 'conf', 'top', 'left', 'right', 'bottom']
        
        # Create feature dict by concatenating all values
        features_dict = {}
        
        # Add image_id if it exists
        if 'image_id' in df.columns:
            features_dict['image_id'] = df['image_id']
        else:
            print("Warning: 'image_id' column not found in DataFrame. It will be missing from the features.")

        # Add numeric columns
        for col in numeric_cols:
            if col in df.columns:
                features_dict[col] = df[col].apply(lambda x: float(x) if pd.notna(x) else 0.0)

        # Add subsetted arms pose vector dimensions (0-22 landmarks, xyz flattened).
        if 'arms_subset_vector' in df.columns:
            arms_dim_count = 0
            for value in df['arms_subset_vector']:
                if value is None:
                    continue
                if isinstance(value, np.ndarray):
                    if value.size > 0:
                        arms_dim_count = int(value.size)
                        break
                elif isinstance(value, (list, tuple)):
                    if len(value) > 0:
                        arms_dim_count = int(len(value))
                        break

            if arms_dim_count > 0:
                for dim_idx in range(arms_dim_count):
                    col_name = f"arms_subset_dim_{dim_idx}"
                    features_dict[col_name] = df['arms_subset_vector'].apply(
                        lambda x: float(x[dim_idx]) if isinstance(x, (list, tuple, np.ndarray)) and len(x) > dim_idx else 0.0
                    )
        
        # Add detection features with descriptive column names
        # ALSO add binary "has_object" indicators to prevent all-zero clustering
        for det_col in detection_cols:
            if det_col in df.columns:
                # Add binary indicator: 1.0 if object present, 0.0 if not
                has_obj_col = f"{det_col}_has_object"
                features_dict[has_obj_col] = df[det_col].apply(
                    lambda x: 1.0 if x is not None else 0.0
                )
                
                # Add standard detection fields
                for i, field in enumerate(detection_fields):
                    col_name = f"{det_col}_{field}"
                    features_dict[col_name] = df[det_col].apply(
                        lambda x: self.flatten_object_detections(x)[i] if x is not None else 0.0
                    )
        
        result_df = pd.DataFrame(features_dict)
        print("prepare_features_for_knn result_df columns: ", result_df.columns)
        return result_df
    
    def prepare_features_for_knn_v2(self, df, fit_scaler=False):
        """
        Enhanced version with optional StandardScaler and per-feature-group weighting.
        
        Args:
            df: DataFrame with ObjectFusion features
            fit_scaler: If True, fit new scaler on this data. If False, use existing scaler.
                       Set True for training data, False for new data assignment.
        
        Returns:
            DataFrame with standardized and weighted features, suitable for K-means.
        """
        import pandas as pd
        from sklearn.preprocessing import StandardScaler
        
        # First, get base features using original method
        features_df = self.prepare_features_for_knn(df)
        
        if not self.USE_FEATURE_STANDARDIZATION:
            return features_df
        
        # Separate image_id before scaling
        image_id_col = None
        if 'image_id' in features_df.columns:
            image_id_col = features_df['image_id'].copy()
            features_df = features_df.drop(columns=['image_id'])
        
        # Group columns by feature type for weighted scaling
        face_angle_cols = ['pitch', 'yaw', 'roll']
        class_id_cols = [col for col in features_df.columns if col.endswith('_class_id')]
        confidence_cols = [col for col in features_df.columns if col.endswith('_conf')]
        bbox_cols = [col for col in features_df.columns if col.endswith(('_top', '_left', '_right', '_bottom'))]
        has_object_cols = [col for col in features_df.columns if col.endswith('_has_object')]
        
        # CRITICAL: Extract class_id columns BEFORE standardization (they're categorical, not continuous)
        class_id_values = features_df[class_id_cols].copy()
        
        # Remove class_id from features to be standardized
        features_to_scale = features_df.drop(columns=class_id_cols)
        
        # Standardize only continuous features (mean=0, std=1)
        if fit_scaler or self.feature_scaler is None:
            self.feature_scaler = StandardScaler()
            scaled_features = self.feature_scaler.fit_transform(features_to_scale)
            if self.VERBOSE:
                print("Fitted new StandardScaler on features (excluding class_id)")
                print(f"  Feature means: {self.feature_scaler.mean_[:5]}...")
                print(f"  Feature stds: {self.feature_scaler.scale_[:5]}...")
        else:
            scaled_features = self.feature_scaler.transform(features_to_scale)
        
        # Convert back to DataFrame to apply per-group weights
        scaled_df = pd.DataFrame(scaled_features, columns=features_to_scale.columns, index=features_df.index)
        
        # Re-insert class_id columns at their ORIGINAL scale (not standardized)
        for col in class_id_cols:
            scaled_df[col] = class_id_values[col]
        
        # Apply feature-group-specific weights
        for col in scaled_df.columns:
            if col in face_angle_cols:
                scaled_df[col] *= self.FEATURE_WEIGHTS['face_angle']
            elif col in class_id_cols:
                scaled_df[col] *= self.FEATURE_WEIGHTS['class_id']
            elif col in confidence_cols:
                scaled_df[col] *= self.FEATURE_WEIGHTS['confidence']
            elif col in has_object_cols:
                scaled_df[col] *= self.FEATURE_WEIGHTS['has_object']
            elif col in bbox_cols:
                scaled_df[col] *= self.FEATURE_WEIGHTS['bbox']
        
        # Re-add image_id if it existed
        if image_id_col is not None:
            scaled_df.insert(0, 'image_id', image_id_col)
        
        if self.VERBOSE:
            print("Feature standardization and weighting applied:")
            print(f"  Face angles: {self.FEATURE_WEIGHTS['face_angle']}x")
            print(f"  Class IDs: {self.FEATURE_WEIGHTS['class_id']}x")
            print(f"  Has Object indicators: {self.FEATURE_WEIGHTS['has_object']}x")
            print(f"  Confidence: {self.FEATURE_WEIGHTS['confidence']}x")
            print(f"  BBox coords: {self.FEATURE_WEIGHTS['bbox']}x")
            print(f"  Final feature range: [{scaled_df.min().min():.2f}, {scaled_df.max().max():.2f}]")
        
        return scaled_df

    def construct_fusion_list(self, row):
        """
        Construct fusion list from a dataframe row for ObjectFusion sorting.
        Returns list format: [pitch, yaw, roll, left_hand(6), right_hand(6), top_face(6),
                              left_eye(6), right_eye(6), mouth(6), shoulder(6),
                              waist(6), feet(6)]
        Total: 57 elements (3 + 9*6)
        """
        fusion_list = row['pitch_yaw_roll_list'].copy()  # Start with [pitch, yaw, roll]
        
        # Add detection data (each is 6 elements: class_id, conf, top, left, right, bottom)
        for col in ['left_hand_object', 'right_hand_object', 'top_face_object',
                    'left_eye_object', 'right_eye_object', 'mouth_object', 'shoulder_object',
                    'waist_object', 'feet_object']:
            detection = row[col]
            if detection is None:
                fusion_list.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # 6 zeros for None
            else:
                fusion_list.extend(self.flatten_object_detections(detection))
        
        return fusion_list

    # ================================================================================
    # REPROCESSING UTILITY METHODS (non-debugging)
    # ================================================================================

    def iter_chunks(self, items, size):
        """Yield successive chunks of specified size from items list."""
        for idx in range(0, len(items), size):
            yield items[idx:idx + size]

    def get_reprocess_image_ids_for_subset(self, engine, image_ids, cutoff_detection_id, chunk_size):
        """
        For a subset of image_ids, return only those with any Detections row where detection_id >= cutoff
        and where ImagesDetections has not yet been marked as processed through that detection_id.
        """
        if not image_ids:
            return set()

        unique_ids = sorted({int(image_id) for image_id in image_ids if pd.notna(image_id)})
        matched_ids = set()
        with engine.connect() as conn:
            for image_id_chunk in self.iter_chunks(unique_ids, chunk_size):
                ids_sql = ",".join(str(image_id) for image_id in image_id_chunk)
                chunk_sql = text(
                    f"""
                    SELECT DISTINCT d.image_id
                    FROM Detections d
                                        LEFT JOIN (
                                                SELECT idt.image_id,
                                                             MAX(idt.last_reprocessed_detection_id) AS last_reprocessed_detection_id
                                                FROM ImagesDetections idt
                                                WHERE idt.image_id IN ({ids_sql})
                                                GROUP BY idt.image_id
                                        ) idt ON idt.image_id = d.image_id
                    WHERE d.detection_id >= :cutoff
                      AND d.image_id IN ({ids_sql})
                                            AND (
                                                    idt.last_reprocessed_detection_id IS NULL
                                                    OR idt.last_reprocessed_detection_id < d.detection_id
                                            )
                    """
                )
                chunk_rows = conn.execute(chunk_sql, {"cutoff": int(cutoff_detection_id)}).fetchall()
                matched_ids.update(int(row[0]) for row in chunk_rows)
        return matched_ids

    def count_existing_images_detections_rows(self, engine, image_ids, chunk_size):
        """
        Count unique image_ids that currently exist in ImagesDetections for a provided id subset.
        """
        if not image_ids:
            return 0

        total_existing = 0
        unique_ids = sorted({int(image_id) for image_id in image_ids if pd.notna(image_id)})
        with engine.connect() as conn:
            for image_id_chunk in self.iter_chunks(unique_ids, chunk_size):
                ids_sql = ",".join(str(image_id) for image_id in image_id_chunk)
                chunk_sql = text(
                    f"""
                    SELECT COUNT(*)
                    FROM ImagesDetections idt
                    WHERE idt.image_id IN ({ids_sql})
                    """
                )
                total_existing += int(conn.execute(chunk_sql).scalar() or 0)
        return total_existing

    def delete_images_detections_rows_for_image_ids(self, engine, image_ids, chunk_size):
        """
        Delete existing ImagesDetections rows for specific image_ids in chunks.
        """
        if not image_ids:
            return 0

        unique_ids = sorted({int(image_id) for image_id in image_ids if pd.notna(image_id)})
        deleted_total = 0
        with engine.begin() as conn:
            for image_id_chunk in self.iter_chunks(unique_ids, chunk_size):
                ids_sql = ",".join(str(image_id) for image_id in image_id_chunk)
                delete_sql = text(f"DELETE FROM ImagesDetections WHERE image_id IN ({ids_sql})")
                result = conn.execute(delete_sql)
                if result.rowcount and result.rowcount > 0:
                    deleted_total += int(result.rowcount)
        return deleted_total

    @staticmethod
    def store_image_face_data(session, target_image_id, face_height=None, nose_pixel_x=None, nose_pixel_y=None, image_h=None, image_w=None, testing=False, auto_commit=True):
        """
        Persist face-height/nose-pixel and image dimensions without overwriting existing values.
        Only NULL fields are populated.
        """
        updated = {
            'encodings': False,
            'images': False,
        }

        def _to_int_or_none(value):
            if value is None:
                return None
            try:
                if pd.isna(value):
                    return None
            except Exception:
                pass
            try:
                return int(value)
            except (TypeError, ValueError, OverflowError):
                return None

        target_image_id = _to_int_or_none(target_image_id)
        if target_image_id is None:
            return updated

        face_height = _to_int_or_none(face_height)
        nose_pixel_x = _to_int_or_none(nose_pixel_x)
        nose_pixel_y = _to_int_or_none(nose_pixel_y)
        image_h = _to_int_or_none(image_h)
        image_w = _to_int_or_none(image_w)

        enc_q = session.query(Encodings).filter(Encodings.image_id == target_image_id)
        if face_height is not None:
            rows = enc_q.filter(Encodings.face_height.is_(None)).update(
                {Encodings.face_height: face_height}, synchronize_session=False
            )
            if rows:
                updated['encodings'] = True
        if nose_pixel_x is not None:
            rows = enc_q.filter(Encodings.nose_pixel_x.is_(None)).update(
                {Encodings.nose_pixel_x: nose_pixel_x}, synchronize_session=False
            )
            if rows:
                updated['encodings'] = True
        if nose_pixel_y is not None:
            rows = enc_q.filter(Encodings.nose_pixel_y.is_(None)).update(
                {Encodings.nose_pixel_y: nose_pixel_y}, synchronize_session=False
            )
            if rows:
                updated['encodings'] = True

        img_q = session.query(Images).filter(Images.image_id == target_image_id)
        if image_h is not None:
            rows = img_q.filter(Images.h.is_(None)).update(
                {Images.h: image_h}, synchronize_session=False
            )
            if rows:
                updated['images'] = True
        if image_w is not None:
            rows = img_q.filter(Images.w.is_(None)).update(
                {Images.w: image_w}, synchronize_session=False
            )
            if rows:
                updated['images'] = True

        if auto_commit and (not testing) and (updated['encodings'] or updated['images']):
            session.commit()

        return updated

    # ================================================================================
    # DEBUG / TRACKING METHODS (disabled when VERBOSE=False)
    # ================================================================================

    def count_tracked_detections_for_image_ids(self, engine, image_ids, class_ids=None, conf_threshold=None, chunk_size=1000):
        """
        Count detections per tracked class for a provided image_id subset.
        Returns dict[class_id] -> {'det_rows': int, 'image_rows': int}.
        Debug method - only outputs if self.VERBOSE is True.
        """
        if class_ids is None:
            class_ids = self.DEBUG_TRACKED_CLASS_IDS
        
        counts = {
            int(class_id): {'det_rows': 0, 'image_rows': 0}
            for class_id in class_ids
        }
        if not image_ids:
            return counts

        unique_ids = sorted({int(image_id) for image_id in image_ids if pd.notna(image_id)})
        if not unique_ids:
            return counts

        class_ids_sql = ",".join(str(int(class_id)) for class_id in sorted(class_ids))
        conf_filter_sql = ""
        params = {}
        if conf_threshold is not None:
            conf_filter_sql = " AND d.conf >= :conf_threshold "
            params['conf_threshold'] = float(conf_threshold)

        with engine.connect() as conn:
            for image_id_chunk in self.iter_chunks(unique_ids, chunk_size):
                ids_sql = ",".join(str(image_id) for image_id in image_id_chunk)
                chunk_sql = text(
                    f"""
                    SELECT
                        d.class_id AS class_id,
                        COUNT(*) AS det_rows,
                        COUNT(DISTINCT d.image_id) AS image_rows
                    FROM Detections d
                    WHERE d.image_id IN ({ids_sql})
                      AND d.class_id IN ({class_ids_sql})
                      AND d.bbox_norm IS NOT NULL
                      AND JSON_EXTRACT(d.bbox_norm, '$.left') IS NOT NULL
                      {conf_filter_sql}
                    GROUP BY d.class_id
                    """
                )
                rows = conn.execute(chunk_sql, params).mappings().all()
                for row in rows:
                    class_id = int(row['class_id'])
                    if class_id in counts:
                        counts[class_id]['det_rows'] += int(row['det_rows'] or 0)
                        counts[class_id]['image_rows'] += int(row['image_rows'] or 0)

        return counts

    def print_reprocessing_dry_run_counts(self, engine, selected_df, cutoff_detection_id, chunk_size):
        """
        Print dry-run counts for reprocessing scope without writing any data.
        Debug method - only outputs if self.VERBOSE is True.
        """
        if not self.VERBOSE:
            return
        
        if selected_df is None or len(selected_df) == 0:
            print("[REPROCESS DRY-RUN] No selected rows available.")
            return

        selected_image_ids = selected_df['image_id'].dropna().astype(int).tolist()
        selected_unique_count = len(set(selected_image_ids))

        with engine.connect() as conn:
            candidate_total_sql = text("""
                SELECT COUNT(DISTINCT d.image_id)
                FROM Detections d
                WHERE d.detection_id >= :cutoff
            """)
            candidate_images_total = int(conn.execute(candidate_total_sql, {
                "cutoff": int(cutoff_detection_id)
            }).scalar() or 0)

            affected_existing_global_sql = text("""
                SELECT COUNT(DISTINCT d.image_id)
                FROM Detections d
                INNER JOIN ImagesDetections idt ON idt.image_id = d.image_id
                WHERE d.detection_id >= :cutoff
            """)
            affected_existing_global = int(conn.execute(affected_existing_global_sql, {
                "cutoff": int(cutoff_detection_id)
            }).scalar() or 0)

        selected_reprocess_ids = self.get_reprocess_image_ids_for_subset(
            engine, selected_image_ids, cutoff_detection_id, chunk_size
        )
        expected_recompute_rows = len(selected_reprocess_ids)
        affected_existing_selected = self.count_existing_images_detections_rows(engine, selected_reprocess_ids, chunk_size)

        print("\n" + "=" * 70)
        print("REPROCESS DRY-RUN SUMMARY (NO WRITES)")
        print("=" * 70)
        print(f"Cutoff detection_id (inclusive): {cutoff_detection_id}")
        print(f"Selected rows from main query: {len(selected_df):,}")
        print(f"Selected unique image_ids: {selected_unique_count:,}")
        print(f"Candidate images (global, Detections cutoff): {candidate_images_total:,}")
        print(f"Affected existing rows (global, join to ImagesDetections): {affected_existing_global:,}")
        print(f"Expected recompute rows (selected run scope): {expected_recompute_rows:,}")
        print(f"Affected existing rows (selected run scope): {affected_existing_selected:,}")
        print("=" * 70 + "\n")
