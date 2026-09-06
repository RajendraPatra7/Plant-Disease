"""Unit tests for the plant disease severity analysis service.

Run from the repo root:

    python -m unittest discover -s backend/tests -v

The tests use only synthetic images (green "leaf" rectangles on a white
background, with brown "lesion" rectangles) so that the expected affected area
is known exactly. No TensorFlow model is required.
"""

import io
import os
import sys
import unittest

import numpy as np
from PIL import Image, ImageDraw

BACKEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
for _p in (os.path.abspath(BACKEND_DIR), os.path.abspath(os.path.join(BACKEND_DIR, ".."))):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from app.services.severity_service import (
        SEVERITY_THRESHOLDS,
        compute_severity,
        estimate_affected_area,
        severity_label_for_score,
    )
except ImportError:
    from backend.app.services.severity_service import (
        SEVERITY_THRESHOLDS,
        compute_severity,
        estimate_affected_area,
        severity_label_for_score,
    )


# Leaf rectangle is drawn as (28, 28, 228, 228) — PIL fills inclusively, so the
# leaf is 201 x 201 = 40401 px.
LEAF_BOX = (28, 28, 228, 228)
LEAF_AREA = (LEAF_BOX[2] - LEAF_BOX[0] + 1) * (LEAF_BOX[3] - LEAF_BOX[1] + 1)
GREEN = (34, 139, 34)     # hue 120 deg -> healthy-green band
BROWN = (139, 69, 19)     # hue  25 deg -> necrotic/brown band
WHITE = (255, 255, 255)


def make_leaf_image(lesions=None):
    """A green leaf rectangle on a white background.

    ``lesions`` is a list of (x0, y0, x1, y1) brown rectangles. Returns the
    exact expected affected-area fraction (0-1) alongside the PIL image.
    """
    im = Image.new("RGB", (256, 256), WHITE)
    d = ImageDraw.Draw(im)
    d.rectangle(LEAF_BOX, fill=GREEN)
    affected = 0
    for box in (lesions or []):
        d.rectangle(box, fill=BROWN)
        affected += (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
    expected_fraction = min(1.0, affected / LEAF_AREA)
    return im, expected_fraction


def full_width_lesion(bottom_y):
    """A lesion spanning the full leaf width, ending at row ``bottom_y``.

    Expected affected fraction = (bottom_y - 27) / 201 (brown replaces green).
    """
    return (LEAF_BOX[0], LEAF_BOX[1], LEAF_BOX[2], bottom_y)


class TestSeverityLabelMapping(unittest.TestCase):
    """Threshold table behaves as documented."""

    def test_threshold_table_is_sorted_and_complete(self):
        self.assertEqual(SEVERITY_THRESHOLDS[0][0], 20)
        self.assertEqual(SEVERITY_THRESHOLDS[-1][0], 100)
        for prev, cur in zip(SEVERITY_THRESHOLDS, SEVERITY_THRESHOLDS[1:]):
            self.assertLess(prev[0], cur[0])

    def test_boundary_scores(self):
        self.assertEqual(severity_label_for_score(0)[0], "Very Low")
        self.assertEqual(severity_label_for_score(20)[0], "Very Low")
        self.assertEqual(severity_label_for_score(21)[0], "Mild")
        self.assertEqual(severity_label_for_score(40)[0], "Mild")
        self.assertEqual(severity_label_for_score(41)[0], "Moderate")
        self.assertEqual(severity_label_for_score(60)[0], "Moderate")
        self.assertEqual(severity_label_for_score(61)[0], "Severe")
        self.assertEqual(severity_label_for_score(80)[0], "Severe")
        self.assertEqual(severity_label_for_score(81)[0], "Critical")
        self.assertEqual(severity_label_for_score(100)[0], "Critical")

    def test_out_of_range_scores_are_clamped(self):
        self.assertEqual(severity_label_for_score(-5)[0], "Very Low")
        self.assertEqual(severity_label_for_score(150)[0], "Critical")

    def test_level_slug_is_machine_readable(self):
        self.assertEqual(severity_label_for_score(35)[1], "mild")


class TestAffectedAreaEstimation(unittest.TestCase):
    """The pure image-analysis step returns expected areas."""

    def test_healthy_leaf_no_affected_area(self):
        im, expected = make_leaf_image()
        area = estimate_affected_area(im)
        self.assertTrue(area["available"])
        self.assertAlmostEqual(area["affected_area_percentage"], 0.0, delta=1.0)

    def test_single_lesion_area_is_measured(self):
        # Full-width lesion ending at row 87 -> (87-27)/201 = 29.9% affected
        im, expected = make_leaf_image([full_width_lesion(87)])
        area = estimate_affected_area(im)
        self.assertAlmostEqual(area["affected_area_percentage"], expected * 100, delta=2.0)

    def test_large_lesion_area_is_measured(self):
        # Full-width lesion ending at row 208 -> 90% affected
        im, expected = make_leaf_image([full_width_lesion(208)])
        area = estimate_affected_area(im)
        self.assertAlmostEqual(area["affected_area_percentage"], expected * 100, delta=2.0)

    def test_foliage_ratio_is_reported(self):
        im, _ = make_leaf_image()
        area = estimate_affected_area(im)
        self.assertGreater(area["foliage_ratio"], 0.5)  # leaf dominates the image

    def test_multiple_regions_are_counted(self):
        im, _ = make_leaf_image([
            (60, 60, 100, 100),
            (150, 150, 190, 190),
            (150, 60, 170, 80),
        ])
        area = estimate_affected_area(im)
        self.assertGreaterEqual(area["affected_regions"], 2)
        self.assertGreater(area["largest_region_fraction"], 0.0)
        self.assertLessEqual(area["largest_region_fraction"], 1.0)

    def test_no_foliage_returns_unavailable(self):
        blank = Image.new("RGB", (256, 256), WHITE)
        area = estimate_affected_area(blank)
        self.assertFalse(area["available"])

    def test_grayscale_image_has_no_foliage(self):
        # A uniform gray image converts to RGB without crashing, but carries no
        # color information, so severity must report "unavailable" rather than
        # fabricate an affected area.
        gray = Image.new("L", (64, 64), 200)
        area = estimate_affected_area(gray)
        self.assertFalse(area["available"])


class TestSeverityComputation(unittest.TestCase):
    """The orchestrator combines image + model info into a severity result."""

    def _severity(self, lesions, status="Diseased", confidence=0.95, margin=0.4):
        im, _ = make_leaf_image(lesions)
        return compute_severity(im, status=status, confidence=confidence, margin=margin, disease="Test Disease")

    def test_result_structure(self):
        sev = self._severity([full_width_lesion(87)])
        for key in ("available", "score", "label", "level", "affected_area_percentage",
                    "affected_regions", "largest_region_fraction", "confidence", "explanation"):
            self.assertIn(key, sev, f"missing key {key}")
        self.assertTrue(sev["available"])
        self.assertIsInstance(sev["score"], int)
        self.assertGreaterEqual(sev["score"], 0)
        self.assertLessEqual(sev["score"], 100)
        self.assertGreaterEqual(sev["confidence"], 0.0)
        self.assertLessEqual(sev["confidence"], 1.0)
        self.assertTrue(sev["explanation"])

    def test_very_low_severity(self):
        sev = self._severity([(60, 60, 100, 100)])  # ~8% affected
        self.assertEqual(sev["label"], "Very Low")
        self.assertEqual(sev["level"], "very_low")

    def test_mild_severity(self):
        sev = self._severity([full_width_lesion(87)])  # ~30% affected
        self.assertEqual(sev["label"], "Mild")
        self.assertGreaterEqual(sev["score"], 21)
        self.assertLessEqual(sev["score"], 40)

    def test_moderate_severity(self):
        sev = self._severity([full_width_lesion(127)])  # ~50% affected
        self.assertEqual(sev["label"], "Moderate")
        self.assertGreaterEqual(sev["score"], 41)
        self.assertLessEqual(sev["score"], 60)

    def test_severe_severity(self):
        sev = self._severity([full_width_lesion(168)])  # ~70% affected
        self.assertEqual(sev["label"], "Severe")
        self.assertGreaterEqual(sev["score"], 61)
        self.assertLessEqual(sev["score"], 80)

    def test_critical_severity(self):
        sev = self._severity([full_width_lesion(208)])  # ~90% affected
        self.assertEqual(sev["label"], "Critical")
        self.assertGreaterEqual(sev["score"], 81)

    def test_healthy_leaf_is_discounted(self):
        # 30% non-green area, but the model says Healthy -> score is heavily
        # reduced and never reported as a dangerous level.
        im, _ = make_leaf_image([full_width_lesion(87)])
        sev = compute_severity(im, status="Healthy", confidence=0.95, margin=0.4)
        self.assertLessEqual(sev["score"], 20)
        self.assertEqual(sev["label"], "Very Low")
        self.assertIn("Healthy", sev["explanation"])

    def test_diseased_leaf_keeps_measured_area(self):
        # Same image, Diseased verdict -> area reported directly (not discounted)
        im, _ = make_leaf_image([full_width_lesion(87)])
        sev = compute_severity(im, status="Diseased", confidence=0.95, margin=0.4)
        self.assertGreater(sev["score"], 20)
        self.assertNotEqual(sev["label"], "Very Low")

    def test_low_confidence_lowers_severity_confidence(self):
        im, _ = make_leaf_image([full_width_lesion(87)])
        high = compute_severity(im, status="Diseased", confidence=0.98, margin=0.5)
        low = compute_severity(im, status="Diseased", confidence=0.66, margin=0.12)
        self.assertLess(low["confidence"], high["confidence"])

    def test_blank_image_is_unavailable(self):
        blank = Image.new("RGB", (256, 256), WHITE)
        sev = compute_severity(blank, status="Diseased", confidence=0.95, margin=0.4)
        self.assertFalse(sev["available"])
        self.assertIn("explanation", sev)

    def test_unreadable_image_is_unavailable(self):
        class NotAnImage:
            def convert(self, *a, **k):
                raise ValueError("not an image")
        sev = compute_severity(NotAnImage(), status="Diseased", confidence=0.95, margin=0.4)
        self.assertFalse(sev["available"])

    def test_diseased_but_no_visible_symptoms_is_honest(self):
        # Model says Diseased but the leaf is pure green -> must not fabricate
        # a dangerous score; explanation should flag subtle symptoms.
        im, _ = make_leaf_image()
        sev = compute_severity(im, status="Diseased", confidence=0.9, margin=0.3)
        self.assertEqual(sev["score"], 0)
        self.assertIn("no strongly discolored regions", sev["explanation"])


class TestApiResponseCompatibility(unittest.TestCase):
    """predict() keeps all existing fields and adds 'severity'."""

    def test_predict_response_shape(self):
        # Import the real service but stub the TensorFlow model so no TF is
        # needed to exercise the full predict() path.
        try:
            import model_service
        except ImportError:
            import backend.app.services.model_service as model_service

        class StubModel:
            def __init__(self, probs):
                self.probs = np.array([probs], dtype=np.float32)
            def predict(self, arr):
                return self.probs

        svc = model_service.ModelService()
        svc.model = StubModel(_one_hot(2))  # Apple Leaf - Cedar Rust

        im, _ = make_leaf_image([full_width_lesion(87)])
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        result = svc.predict(buf.getvalue())

        # Existing fields all still present
        for key in ("is_valid_leaf", "class_index", "full_class_name", "crop",
                    "disease", "status", "confidence", "confidence_percentage",
                    "recommendations"):
            self.assertIn(key, result)
        self.assertTrue(result["is_valid_leaf"])

        # New severity field present and structured
        self.assertIn("severity", result)
        self.assertTrue(result["severity"]["available"])
        self.assertEqual(result["severity"]["label"], "Mild")

    def test_invalid_leaf_response_unchanged(self):
        try:
            import model_service
        except ImportError:
            import backend.app.services.model_service as model_service

        class StubModel:
            def __init__(self, probs):
                self.probs = np.array([probs], dtype=np.float32)
            def predict(self, arr):
                return self.probs

        svc = model_service.ModelService()
        svc.model = StubModel(_one_hot(2))

        buf = io.BytesIO()
        Image.new("RGB", (256, 256), (220, 30, 30)).save(buf, format="PNG")  # red, no foliage
        result = svc.predict(buf.getvalue())

        self.assertFalse(result["is_valid_leaf"])
        self.assertIn("error_message", result)
        self.assertNotIn("severity", result)  # severity not meaningful on a non-leaf


def _one_hot(winner_idx, top1=0.94, top2=0.03):
    """A plausible 38-class softmax vector with a clear winner."""
    probs = np.full(38, 0.0008, dtype=np.float32)
    probs[winner_idx] = top1
    probs[(winner_idx + 1) % 38] = top2
    return probs


if __name__ == "__main__":
    unittest.main()
