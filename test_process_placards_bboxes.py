import ast
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

from tools_yolo import YOLOTools


def load_detection_functions():
    # Importing process_placards initializes models and a live database session.
    path = Path(__file__).with_name("process_placards.py")
    tree = ast.parse(path.read_text())
    names = {
        "_normalize_bbox_for_image",
        "bbox_to_cluster_id",
        "assign_hsv_detect_results",
        "filter_invalid_detection_bboxes",
        "do_yolo_detections",
    }
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {"json": json}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


class PlacardBoundingBoxTests(unittest.TestCase):
    def setUp(self):
        self.functions = load_detection_functions()
        yolo = YOLOTools.__new__(YOLOTools)
        yolo.io = SimpleNamespace(unstring_json=lambda value: json.loads(value) if isinstance(value, str) else value)
        yolo.save_obj_bbox = Mock()
        self.yolo = yolo
        self.save_no_detections = Mock()
        self.delete_no_detections = Mock()
        self.save_ids = Mock()
        self.functions.update({
            "io": yolo.io,
            "yolo": yolo,
            "cl": SimpleNamespace(prep_pose_clusters_enc=lambda hsl: (39, 0.1)),
            "meta_cluster_dict": {39: 3},
            "NoDetections": object(),
            "NoDetectionsCustom": object(),
            "Detections": object(),
            "yolo_model": object(),
            "yolo_custom_model": object(),
            "custom_ids_to_global_dict": {34: 80},
            "IOU_THRESHOLD": 0.5,
            "ADJACENCY_THRESHOLD_PX": 20,
            "VERBOSE": False,
            "DEBUGGING": False,
            "SAVE_NEW_LABELS": False,
            "TESTING_NO_DB_WRITE": False,
            "session": object(),
            "save_no_dectections": self.save_no_detections,
            "delete_no_detections": self.delete_no_detections,
            "save_new_detection_ids_for_reprocess": self.save_ids,
        })
        self.image = np.zeros((866, 1288, 3), dtype=np.uint8)
        self.result = SimpleNamespace(image_id=3469535, imagename="example.jpg")

    @staticmethod
    def detection(bbox, class_id=34, obj_no=1):
        return {"class_id": class_id, "obj_no": obj_no, "bbox": json.dumps(bbox), "conf": 0.73}

    def run_detections(self, detections, custom=True):
        return self.functions["do_yolo_detections"](
            self.result, self.image, "example.jpg", [], custom=custom,
            precomputed_raw_detections=detections, do_commit=False,
        )

    def test_reported_zero_height_box_is_skipped_without_false_no_detection(self):
        box = {"left": 0, "top": 866, "right": 1288, "bottom": 866}
        self.functions.update({"DEBUGGING": True, "SAVE_NEW_LABELS": True})
        self.assertEqual(self.run_detections([self.detection(box)]), [])
        self.save_no_detections.assert_not_called()
        self.yolo.save_obj_bbox.assert_not_called()

    def test_inverted_or_zero_width_boxes_are_rejected(self):
        boxes = [
            {"left": 5, "top": 0, "right": 5, "bottom": 10},
            {"left": 10, "top": 10, "right": 5, "bottom": 20},
        ]
        self.assertEqual(self.run_detections([self.detection(box) for box in boxes]), [])
        self.save_no_detections.assert_not_called()

    def test_invalid_box_does_not_merge_with_or_discard_valid_detection(self):
        invalid = self.detection({"left": 0, "top": 866, "right": 1288, "bottom": 866})
        valid = self.detection({"left": 0, "top": 850, "right": 100, "bottom": 866}, obj_no=2)
        results = self.run_detections([invalid, valid])
        self.assertEqual(len(results), 1)
        self.assertEqual(json.loads(results[0]["bbox"]), json.loads(valid["bbox"]))
        self.assertEqual((results[0]["class_id"], results[0]["cluster_id"]), (80, 39))
        self.assertTrue(results[0]["hsv_redone"])
        self.yolo.save_obj_bbox.assert_called_once()
        self.delete_no_detections.assert_called_once()

    def test_out_of_bounds_boxes_are_clipped_or_rejected(self):
        partial = self.detection({"left": -3, "top": 864, "right": 20, "bottom": 900})
        outside = self.detection({"left": 1290, "top": 0, "right": 1300, "bottom": 30}, obj_no=2)
        results = self.run_detections([partial, outside])
        self.assertEqual(json.loads(results[0]["bbox"]), {"left": 0, "top": 864, "right": 20, "bottom": 866})

    def test_no_detections_and_ignored_person_still_record_no_detection(self):
        self.run_detections([], custom=False)
        self.save_no_detections.assert_called_once()
        self.save_no_detections.reset_mock()
        person = self.detection({"left": 0, "top": 866, "right": 10, "bottom": 866}, class_id=0)
        self.run_detections([person], custom=False)
        self.save_no_detections.assert_called_once()

    def test_nonfinite_coordinates_are_rejected(self):
        box = {"left": 0, "top": 0, "right": float("inf"), "bottom": 10}
        self.assertEqual(self.run_detections([self.detection(box)]), [])
        self.save_no_detections.assert_not_called()


if __name__ == "__main__":
    unittest.main()
