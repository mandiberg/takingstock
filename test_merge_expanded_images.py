import contextlib
import io
import os
import tempfile
import unittest
import weakref
from unittest import mock

import cv2
import numpy as np

import merge_expanded_images as merge


class StillMergeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.settings = mock.patch.multiple(
            merge,
            STILL_MERGE_TEMP_DIR=self.temp.name,
            IS_VIDEO=False,
            IS_VIDEO_MERGE=False,
            USE_CANONICAL_RATIOS=False,
            CUTOFF_THRESHOLD=0,
            USE_THRESHOLD_FOR_IMAGES=True,
        )
        self.settings.start()
        self.addCleanup(self.settings.stop)
        self.output = contextlib.redirect_stdout(io.StringIO())
        self.output.__enter__()
        self.addCleanup(self.output.__exit__, None, None, None)

    def make_images(self, count):
        rng = np.random.default_rng(42)
        paths = []
        for i in range(count):
            image = rng.integers(0, 256, (8 + i % 2, 9 + i % 3, 3), dtype=np.uint8)
            path = os.path.join(self.temp.name, f"source-{i}.png")
            self.assertTrue(cv2.imwrite(path, image))
            paths.append(path)
        return paths

    def legacy_merge(self, paths, output_dims):
        images = merge.load_images(paths, self.temp.name)
        successes = 0
        while len(images) >= 2:
            images, successes = merge.iterate_image_list(self.temp.name, images, successes, output_dims)
        return images[0], successes

    def test_matches_legacy_pairwise_rounds_for_all_counts(self):
        paths = self.make_images(64)
        for count in range(2, 65):
            with self.subTest(count=count):
                expected, expected_count = self.legacy_merge(paths[:count], (6, 7))
                actual, actual_count, readable = merge.merge_still_streaming(paths[:count], self.temp.name, (6, 7))
                np.testing.assert_array_equal(actual, expected)
                self.assertEqual(actual_count, expected_count)
                self.assertEqual(readable, count)
        self.assertFalse(any(name.startswith("takingstock-merge-") for name in os.listdir(self.temp.name)))

    def test_unreadable_inputs_and_threshold(self):
        paths = self.make_images(5)
        paths.insert(1, os.path.join(self.temp.name, "missing.png"))
        expected, expected_count = self.legacy_merge(paths, (6, 7))
        actual, actual_count, readable = merge.merge_still_streaming(paths, self.temp.name, (6, 7))
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual((actual_count, readable), (expected_count, 5))
        with mock.patch.object(merge, "CUTOFF_THRESHOLD", 6):
            actual, actual_count, readable = merge.merge_still_streaming(paths, self.temp.name, (6, 7))
        self.assertIsNone(actual)
        self.assertEqual((actual_count, readable), (0, 5))

    def test_limit_and_source_images_are_released(self):
        paths = self.make_images(66)
        original_read = merge.safe_imread
        references = []

        def tracked_read(path):
            self.assertFalse(any(ref() is not None for ref in references))
            image = original_read(path)
            references.append(weakref.ref(image))
            return image

        with mock.patch.object(merge, "safe_imread", side_effect=tracked_read):
            actual, count, readable = merge.merge_still_streaming(paths, self.temp.name, (6, 7))
        self.assertEqual((count, readable, len(references)), (63, 64, 64))
        self.assertEqual(actual.shape, (6, 7, 3))

    def test_canonical_preprocessing_matches_legacy(self):
        paths = self.make_images(7)
        with mock.patch.object(merge, "USE_CANONICAL_RATIOS", True):
            expected, expected_count = self.legacy_merge(paths, (6, 7))
            actual, actual_count, readable = merge.merge_still_streaming(paths, self.temp.name, (6, 7))
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual((actual_count, readable), (expected_count, 7))

    def test_temporary_files_removed_after_merge_error(self):
        paths = self.make_images(2)
        with mock.patch.object(merge, "iterate_image_list", side_effect=RuntimeError("merge failed")):
            with self.assertRaisesRegex(RuntimeError, "merge failed"):
                merge.merge_still_streaming(paths, self.temp.name, (6, 7))
        self.assertFalse(any(name.startswith("takingstock-merge-") for name in os.listdir(self.temp.name)))

    def test_worker_limits_paths_before_dimension_sampling(self):
        paths = self.make_images(66)
        with mock.patch.object(merge, "get_cluster_input_paths", return_value=paths), \
             mock.patch.object(merge, "get_median_image_dimensions", return_value=(6, 7)) as dimensions, \
             mock.patch.object(merge, "save_merge") as save:
            result = merge._still_merge_worker(self.temp.name)
        self.assertTrue(result["success"])
        self.assertEqual(dimensions.call_args.args[0], paths[:64])
        self.assertEqual(save.call_args.args[0].shape, (6, 7, 3))

    def test_write_failure_is_reported(self):
        with mock.patch.object(merge.cv2, "imwrite", return_value=False):
            with self.assertRaises(OSError):
                merge.save_merge(np.zeros((1, 1, 3), dtype=np.uint8), 1, None, None, None, None, None, self.temp.name)


if __name__ == "__main__":
    unittest.main()
