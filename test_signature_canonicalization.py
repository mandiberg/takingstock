import unittest

import numpy as np
import pandas as pd

from tools_clustering import ToolsClustering


class TestSignatureCanonicalization(unittest.TestCase):
    def test_bicycle_handedness_collapses_to_one_signature(self):
        cl = ToolsClustering("object_fusion")

        left = cl._build_token_from_slot_labels({
            "LH": 1, "RH": 0, "TF": 0, "LE": 0, "RE": 0, "MO": 0, "SH": 0, "WA": 0, "FT": 0,
        })
        right = cl._build_token_from_slot_labels({
            "LH": 0, "RH": 1, "TF": 0, "LE": 0, "RE": 0, "MO": 0, "SH": 0, "WA": 0, "FT": 0,
        })
        both = cl._build_token_from_slot_labels({
            "LH": 1, "RH": 1, "TF": 0, "LE": 0, "RE": 0, "MO": 0, "SH": 0, "WA": 0, "FT": 0,
        })

        self.assertEqual(left[1], right[1])
        self.assertEqual(right[1], both[1])

    def test_handheld_weapon_classes_canonicalize_to_bicycle_in_hand_and_drop_lower_body(self):
        cl = ToolsClustering("object_fusion")

        bike = cl._build_token_from_slot_labels({
            "LH": 1, "RH": 0, "TF": 0, "LE": 0, "RE": 0, "MO": 0, "SH": 0, "WA": 0, "FT": 0,
        })
        rifle_hand_bike = cl._build_token_from_slot_labels({
            "LH": 1, "RH": 0, "TF": 0, "LE": 0, "RE": 0, "MO": 0, "SH": 0, "WA": 0, "FT": 109,
        })
        rifle_waist = cl._build_token_from_slot_labels({
            "LH": 0, "RH": 0, "TF": 0, "LE": 0, "RE": 0, "MO": 0, "SH": 0, "WA": 109, "FT": 0,
        })
        dumbbell_hand_bike = cl._build_token_from_slot_labels({
            "LH": 1, "RH": 0, "TF": 0, "LE": 0, "RE": 0, "MO": 0, "SH": 0, "WA": 0, "FT": 86,
        })

        self.assertEqual(rifle_hand_bike[0], bike[0])
        self.assertEqual(rifle_hand_bike[1], bike[1])
        self.assertEqual(rifle_waist[0], "LH:0|RH:0|TF:0|LE:0|RE:0|MO:0|SH:0|WA:109|FT:0")
        self.assertNotEqual(rifle_waist[1], bike[1])
        self.assertEqual(dumbbell_hand_bike[0], bike[0])
        self.assertEqual(dumbbell_hand_bike[1], bike[1])

    def test_weight_like_classes_do_not_rewrite_without_bike_in_hand(self):
        cl = ToolsClustering("object_fusion")

        rifle_only = cl._build_token_from_slot_labels({
            "LH": 109, "RH": 0, "TF": 0, "LE": 0, "RE": 0, "MO": 0, "SH": 0, "WA": 0, "FT": 0,
        })
        dumbbell_only = cl._build_token_from_slot_labels({
            "LH": 86, "RH": 0, "TF": 0, "LE": 0, "RE": 0, "MO": 0, "SH": 0, "WA": 0, "FT": 0,
        })
        barbell_only = cl._build_token_from_slot_labels({
            "LH": 156, "RH": 0, "TF": 0, "LE": 0, "RE": 0, "MO": 0, "SH": 0, "WA": 0, "FT": 0,
        })

        self.assertNotEqual(rifle_only[0], "LH:1|RH:1|TF:0|LE:0|RE:0|MO:0|SH:0|WA:0|FT:0")
        self.assertNotEqual(dumbbell_only[0], "LH:1|RH:1|TF:0|LE:0|RE:0|MO:0|SH:0|WA:0|FT:0")
        self.assertNotEqual(barbell_only[0], "LH:1|RH:1|TF:0|LE:0|RE:0|MO:0|SH:0|WA:0|FT:0")

    def test_duplicate_backpack_handbag_on_shoulder_and_waist_keeps_shoulder_only(self):
        cl = ToolsClustering("object_fusion")

        shoulder_only = cl._build_token_from_slot_labels({
            "LH": 0, "RH": 0, "TF": 0, "LE": 0, "RE": 0, "MO": 0, "SH": 24, "WA": 0, "FT": 0,
        })
        duped = cl._build_token_from_slot_labels({
            "LH": 0, "RH": 0, "TF": 0, "LE": 0, "RE": 0, "MO": 0, "SH": 24, "WA": 24, "FT": 0,
        })

        self.assertEqual(duped[0], shoulder_only[0])

    def test_replace_anchor_helper_accepts_scalar_replacement_value(self):
        cl = ToolsClustering("object_fusion")
        normalized = {"LH": 155, "RH": 86, "TF": 0, "LE": 0, "RE": 0, "MO": 0, "SH": 0, "WA": 0, "FT": 0}

        result = cl._replace_anchor_class_in_slots(
            normalized,
            anchor_class_id=155,
            slots=("LH", "RH"),
            replace_values=86,
            trigger_slots=("LH", "RH"),
        )

        self.assertEqual(result["LH"], 155)
        self.assertEqual(result["RH"], 155)

    def test_second_pass_canonicalization_after_collapse_rewrites_handheld_noise(self):
        cl = ToolsClustering("object_fusion")

        slots = {
            "LH": 1,
            "RH": 0,
            "TF": 0,
            "LE": 0,
            "RE": 0,
            "MO": 0,
            "SH": 0,
            "WA": 109,
            "FT": 0,
        }

        canonical, _, _, _ = cl.canonicalize_full_signature(slots)

        self.assertEqual(canonical["LH"], 1)
        self.assertEqual(canonical["RH"], 1)
        self.assertEqual(canonical["WA"], 0)
        self.assertEqual(canonical["FT"], 0)
        self.assertEqual(cl._build_token_from_slot_labels(slots)[0], "LH:1|RH:1|TF:0|LE:0|RE:0|MO:0|SH:0|WA:0|FT:0")

    def test_leg_pose_boundary_in_midrange_valley_is_separable(self):
        cl = ToolsClustering("ArmsPoses3D")
        left = np.clip(np.random.default_rng(7).normal(1.0, 0.18, 120), 0.6, 1.5)
        valley = np.clip(np.random.default_rng(11).normal(2.5, 0.14, 10), 2.1, 3.0)
        right = np.clip(np.random.default_rng(13).normal(4.8, 0.22, 120), 4.0, 5.5)

        df = pd.DataFrame({
            "visible_leg_count": np.ones(len(left) + len(valley) + len(right), dtype=int),
            "leg_extension_max": np.concatenate([left, valley, right]),
        })

        result = cl.assess_leg_pose_separability(
            df,
            floor_pct=0.0,
            min_bucket_size=10,
            min_gap_ratio=0.3,
            cluster_label="test-midrange-valley",
        )

        self.assertTrue(result["is_separable"])
        self.assertGreaterEqual(result["boundary"], 1.0)
        self.assertLessEqual(result["boundary"], 5.0)
        self.assertLessEqual(result["gap_ratio"], 1.0)

    def test_leg_pose_labels_distinguish_left_and_right_knee_cross(self):
        cl = ToolsClustering("ArmsPoses3D")

        df = pd.DataFrame({
            "visible_leg_count": [2, 2, 2, 2],
            "leg_extension_max": [0.8, 0.9, 0.7, 0.85],
            "leg_extension_min": [0.1, 0.2, 0.15, 0.18],
            "leg_asymmetry": [0.7, 0.7, 0.55, 0.67],
            "ankle_rel_y_left": [1.4, 0.8, 1.0, 0.9],
            "ankle_rel_y_right": [0.8, 1.4, 0.3, 1.2],
        })

        labels = cl.label_by_leg_pose(df, boundary=1.5)

        self.assertEqual(labels.iloc[0], "folded_left_on_right_knee")
        self.assertEqual(labels.iloc[1], "folded_right_on_left_knee")
        self.assertEqual(labels.iloc[2], "folded_left_on_right_knee")
        self.assertEqual(labels.iloc[3], "folded_right_on_left_knee")

    def test_leg_pose_labels_use_knee_height_when_ankles_are_tied(self):
        cl = ToolsClustering("ArmsPoses3D")

        df = pd.DataFrame({
            "visible_leg_count": [2, 2],
            "leg_extension_max": [0.9, 0.9],
            "leg_extension_min": [0.2, 0.2],
            "leg_asymmetry": [0.3, 0.3],
            "ankle_rel_y_left": [0.78, 0.93],
            "ankle_rel_y_right": [0.8, 0.91],
            "knee_rel_y_left": [1.35, 0.72],
            "knee_rel_y_right": [0.72, 1.33],
        })

        labels = cl.label_by_leg_pose(df, boundary=1.5)

        self.assertEqual(labels.iloc[0], "folded_left_on_right_knee")
        self.assertEqual(labels.iloc[1], "folded_right_on_left_knee")

    def test_leg_pose_labels_detect_standing_raised_leg(self):
        cl = ToolsClustering("ArmsPoses3D")

        df = pd.DataFrame({
            "visible_leg_count": [2, 2],
            "leg_extension_max": [3.0, 2.9],
            "leg_extension_min": [0.2, 0.3],
            "leg_asymmetry": [2.8, 2.6],
            "ankle_rel_y_left": [0.2, 0.9],
            "ankle_rel_y_right": [2.8, 0.3],
            "knee_rel_y_left": [0.8, 1.3],
            "knee_rel_y_right": [1.6, 0.7],
            "mid_hip_x": [0.0, 0.0],
            "knee_left_x": [-0.85, -0.70],
            "knee_right_x": [0.10, 0.80],
            "foot_left_x": [-0.90, -0.80],
            "foot_right_x": [0.05, 0.90],
        })

        labels = cl.label_by_leg_pose(df, boundary=1.5)

        self.assertEqual(labels.iloc[0], "standing_raised_left_leg")
        self.assertEqual(labels.iloc[1], "standing_raised_right_leg")

    def test_leg_pose_multiplier_variant_uses_modal_leg_length(self):
        cl = ToolsClustering("ArmsPoses3D")

        folded_variant = cl.derive_leg_pose_multiplier_variant(
            "folded_left_on_right_knee",
            pd.Series([0.8, 0.9, 1.1, 1.2]),
        )
        standing_variant = cl.derive_leg_pose_multiplier_variant(
            "standing_raised_left_leg",
            pd.Series([4.8, 5.0, 5.1, 4.9]),
        )
        tall_variant = cl.derive_leg_pose_multiplier_variant(
            "standing_raised_left_leg",
            pd.Series([5.5, 5.7, 5.6]),
        )

        self.assertEqual(folded_variant, 1)
        self.assertEqual(standing_variant, 5)
        self.assertEqual(tall_variant, 6)
        self.assertNotEqual(folded_variant, standing_variant)


if __name__ == "__main__":
    unittest.main()
