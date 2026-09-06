"""
Plant Disease Severity Analysis Service
========================================

Estimates how much of an uploaded leaf is *visibly affected* by disease and
combines that with the classifier's verdict into a structured, machine-readable
severity result.

Why this exists
---------------
The deployed CNN (MobileNetV2 fine-tuned on 38 PlantVillage classes) is a pure
image classifier: it outputs only a softmax probability vector and NO pixel
masks, bounding boxes, or affected-area estimates. Therefore there is no
scientific basis for equating severity with classification confidence
("confidence is not severity"). Instead, this module derives the severity
signal from the image itself:

  1. Segment the leaf into color bands using the same HSV thresholds the
     existing leaf validator (``model_service.validate_leaf_image``) relies on:
       - healthy green foliage        (hue 65-165)
       - necrotic / yellow / brown    (hue 20-65)  <- primary symptom signal
       - pale / powdery-mildew white  (low saturation, high value) <- secondary
  2. Affected area = (necrotic pixels + weighted pale-mildew pixels) divided
     by total foliage pixels  ->  a percentage of the *leaf*, not the image.
  3. Label connected symptom regions (scattered spots vs. one dominant lesion)
     for context.
  4. Fold in the model's verdict:
       - "Diseased"  -> severity score = measured affected area
       - "Healthy"   -> measured area is discounted (healthy leaves routinely
                        show veins, edges, sunburn, dirt that color analysis
                        cannot fully separate from disease)
  5. Attach a *severity confidence* (trust in the estimate) that is separate
     from the score and driven by: diagnosis confidence, top-2 margin, how much
     of the image is foliage, and whether symptoms were actually measurable.

Limitations (be honest)
-----------------------
- Affected area is ESTIMATED from color, not measured by the network. Diseases
  whose symptoms are shape-based (leaf curling), very subtle, or that produce
  colors outside the bands above (e.g. early yellow-curl mottling) are
  systematically under-estimated.
- Color segmentation cannot separate true symptoms from sunburn, senescence,
  dirt, or lighting artifacts; the Healthy-class discount and the severity
  confidence are designed to keep those false positives from being misleading.
- Region statistics are computed on a downsampled grid (see REGION_GRID_SIZE)
  and are therefore approximate.

This module deliberately imports only NumPy + Pillow (no TensorFlow) so it can
be unit-tested in isolation and can never fail the model inference path.
"""

import collections

import numpy as np
from PIL import Image


# --------------------------------------------------------------------------
# Configurable severity scale (score is normalized to 0-100)
# Each entry: (max_inclusive_score, human label, machine-readable level)
# --------------------------------------------------------------------------
SEVERITY_THRESHOLDS = [
    (20, "Very Low", "very_low"),
    (40, "Mild", "mild"),
    (60, "Moderate", "moderate"),
    (80, "Severe", "severe"),
    (100, "Critical", "critical"),
]

# A "Healthy" classification discounts the measured affected area by this
# factor. Healthy leaves routinely show a few percent of non-green pixels
# (veins, edges, senescence, specular highlights) that color segmentation
# cannot distinguish from early disease.
HEALTHY_SEVERITY_DISCOUNT = 0.3

# Weight applied to pale/mildew pixels when estimating affected area. The band
# is intentionally under-weighted because specular highlights share its
# low-saturation / high-value signature.
MILDEW_BAND_WEIGHT = 0.5

# Below this affected-area % we treat symptoms as "not measurable" and lower
# the severity confidence accordingly.
MIN_MEASURABLE_AREA_PCT = 0.5

# Foliage coverage of the whole image at which severity confidence saturates.
FOLIAGE_FULL_COVERAGE = 0.5

# Minimum symptom-region size, as a fraction of total foliage, to count as a
# real region rather than sensor/lighting noise.
MIN_REGION_FRACTION = 0.002

# Working resolution for connected-component analysis (keeps labeling fast).
REGION_GRID_SIZE = 64


def rgb_to_hsv_numpy(rgb_array):
    """Vectorized RGB -> HSV in pure NumPy (0-360 Hue, 0-1 Sat, 0-1 Val).

    Mirrors the implementation in ``model_service.py`` so the severity bands
    are consistent with the existing leaf-validator bands.
    """
    rgb = rgb_array.astype(np.float32) / 255.0
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

    max_c = np.max(rgb, axis=2)
    min_c = np.min(rgb, axis=2)
    delta = max_c - min_c

    h = np.zeros_like(max_c)
    mask_delta = delta > 1e-5

    mask_r = mask_delta & (max_c == r)
    h[mask_r] = ((g[mask_r] - b[mask_r]) / (delta[mask_r] + 1e-8)) % 6.0

    mask_g = mask_delta & (max_c == g)
    h[mask_g] = ((b[mask_g] - r[mask_g]) / (delta[mask_g] + 1e-8)) + 2.0

    mask_b = mask_delta & (max_c == b)
    h[mask_b] = ((r[mask_b] - g[mask_b]) / (delta[mask_b] + 1e-8)) + 4.0

    h = (h * 60.0) % 360.0

    s = np.zeros_like(max_c)
    s[max_c > 1e-5] = delta[max_c > 1e-5] / max_c[max_c > 1e-5]
    v = max_c

    return h, s, v


def _connected_component_sizes(binary_mask):
    """Return the pixel sizes of all 4-connected regions in ``binary_mask``.

    Pure Python BFS (no scipy/OpenCV dependency). Intended for the small
    downsampled grid used by :func:`_region_stats`.
    """
    h, w = binary_mask.shape
    visited = np.zeros_like(binary_mask, dtype=bool)
    sizes = []
    queue = collections.deque()

    for y, x in zip(*np.nonzero(binary_mask)):
        if visited[y, x]:
            continue
        # BFS flood fill from this seed pixel
        queue.append((int(y), int(x)))
        visited[y, x] = True
        size = 0
        while queue:
            cy, cx = queue.popleft()
            size += 1
            if cy > 0 and binary_mask[cy - 1, cx] and not visited[cy - 1, cx]:
                visited[cy - 1, cx] = True
                queue.append((cy - 1, cx))
            if cy < h - 1 and binary_mask[cy + 1, cx] and not visited[cy + 1, cx]:
                visited[cy + 1, cx] = True
                queue.append((cy + 1, cx))
            if cx > 0 and binary_mask[cy, cx - 1] and not visited[cy, cx - 1]:
                visited[cy, cx - 1] = True
                queue.append((cy, cx - 1))
            if cx < w - 1 and binary_mask[cy, cx + 1] and not visited[cy, cx + 1]:
                visited[cy, cx + 1] = True
                queue.append((cy, cx + 1))
        sizes.append(size)

    return sizes


def _region_stats(affected_mask, foliage_pixels):
    """Approximate count + concentration of symptom regions.

    Returns ``(region_count, largest_region_fraction)``. The mask is
    downsampled to REGION_GRID_SIZE for speed; ``largest_region_fraction`` is
    the share of all affected pixels contained in the single largest region.
    """
    grid = REGION_GRID_SIZE
    small = Image.fromarray((affected_mask * 255).astype(np.uint8)).resize(
        (grid, grid), Image.NEAREST
    )
    small_mask = np.array(small) > 0

    sizes = _connected_component_sizes(small_mask)
    # Fraction-of-foliage threshold expressed on the downsampled grid. Both the
    # region area and the foliage area scale by the same factor, so the
    # threshold in grid pixels is constant: MIN_REGION_FRACTION * grid^2.
    min_region_px = max(2, int(MIN_REGION_FRACTION * grid * grid))
    sizes = [s for s in sizes if s >= min_region_px]

    if not sizes:
        return 0, 0.0

    total = float(sum(sizes))
    return len(sizes), round(max(sizes) / total, 4)


def estimate_affected_area(pil_image):
    """Pure image-analysis step: measure visible symptoms on the leaf.

    Returns a dict with ``available``, ``foliage_ratio`` (foliage coverage of
    the whole image), ``affected_area_percentage`` (% of *foliage* that shows
    symptoms), ``affected_pixels``, ``foliage_pixels``,
    ``affected_regions`` and ``largest_region_fraction``.
    """
    try:
        img_rgb = np.array(pil_image.convert("RGB"))
    except Exception:
        return {
            "available": False,
            "foliage_ratio": None,
            "affected_area_percentage": None,
            "affected_regions": None,
            "largest_region_fraction": None,
            "explanation": "Image could not be decoded as an RGB array.",
        }
    if img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
        return {
            "available": False,
            "foliage_ratio": None,
            "affected_area_percentage": None,
            "affected_regions": None,
            "largest_region_fraction": None,
            "explanation": "Image could not be read as an RGB array.",
        }

    h, s, v = rgb_to_hsv_numpy(img_rgb)

    # Healthy green chlorophyll spectrum (same band as the leaf validator)
    mask_green = (h >= 65) & (h <= 165) & (s >= 0.15) & (v >= 0.15)
    # Necrotic / yellow / brown / rust spectrum (same "brown" band as validator)
    mask_brown = (h >= 20) & (h < 65) & (s >= 0.15) & (v >= 0.15)
    # Pale powdery-mildew / whitish patch spectrum (low saturation, bright).
    # Hue kept within 15-90 to avoid neutral gray backgrounds and blown-out
    # highlights; v <= 0.95 excludes pure white.
    mask_mildew = (s <= 0.28) & (v >= 0.50) & (v <= 0.95) & (h >= 15) & (h < 90)

    foliage_mask = mask_green | mask_brown | mask_mildew
    foliage_pixels = int(np.sum(foliage_mask))
    total_pixels = float(img_rgb.shape[0] * img_rgb.shape[1])

    if foliage_pixels <= 0:
        return {
            "available": False,
            "foliage_ratio": 0.0,
            "affected_area_percentage": None,
            "affected_regions": None,
            "largest_region_fraction": None,
            "explanation": "No foliage-colored region detected in the image; affected area cannot be estimated.",
        }

    # Weighted symptom count: necrotic/yellow pixels count fully, pale-mildew
    # pixels count at MILDEW_BAND_WEIGHT (they can be specular highlights).
    affected_count = float(
        np.sum(mask_brown) + MILDEW_BAND_WEIGHT * float(np.sum(mask_mildew))
    )
    affected_area_pct = min(100.0, 100.0 * affected_count / float(foliage_pixels))

    affected_mask = mask_brown | mask_mildew
    region_count, largest_fraction = _region_stats(affected_mask, foliage_pixels)

    return {
        "available": True,
        "foliage_ratio": round(foliage_pixels / total_pixels, 4),
        "affected_area_percentage": round(affected_area_pct, 1),
        "affected_pixels": int(round(affected_count)),
        "foliage_pixels": foliage_pixels,
        "affected_regions": region_count,
        "largest_region_fraction": largest_fraction,
    }


def severity_label_for_score(score):
    """Map a normalized 0-100 score to a (label, level) tuple.

    Uses the configurable :data:`SEVERITY_THRESHOLDS` table.
    """
    score = int(round(max(0.0, min(100.0, float(score)))))
    for max_score, label, level in SEVERITY_THRESHOLDS:
        if score <= max_score:
            return label, level
    # Fallback (should be unreachable): top of the scale
    return SEVERITY_THRESHOLDS[-1][1], SEVERITY_THRESHOLDS[-1][2]


def _build_explanation(status, area, regions, disease, measured_note):
    """Compose a short, honest, human-readable summary of the severity result."""
    if status == "Healthy":
        base = (
            f"The model classifies this leaf as Healthy. Measured non-green area "
            f"is {area:.1f}%, which is discounted for healthy leaves (natural "
            f"veins, edges, senescence or dirt cannot be fully separated from "
            f"early symptoms by color analysis)."
        )
        if regions >= 2:
            base += f" Discoloration appears across {regions} separate regions."
        return base

    if area <= MIN_MEASURABLE_AREA_PCT:
        return (
            f"The model detects {disease or 'a disease'}, but no strongly discolored "
            f"regions were found by color analysis. Symptoms may be subtle, "
            f"early-stage, or shape-based (e.g. curling) and are not captured "
            f"by pixel color."
        )

    base = (
        f"An estimated {area:.1f}% of the leaf area shows visible disease "
        f"symptoms (necrotic/yellow tissue or pale mildew)."
    )
    if regions >= 2:
        base += f" Symptoms are spread across {regions} separate regions."
    elif regions == 1:
        base += " Symptoms are concentrated in a single dominant region."
    if measured_note:
        base += f" {measured_note}"
    return base


def compute_severity(pil_image, *, status, confidence, margin, disease=None):
    """Compute the structured severity result for one prediction.

    Parameters
    ----------
    pil_image : PIL.Image
        The (preferably RGB) uploaded image.
    status : {"Healthy", "Diseased"}
        The classifier's verdict.
    confidence : float
        Top-1 softmax probability of the predicted class (0-1).
    margin : float
        Top-1 minus top-2 softmax probabilities (0-1); 0 if unavailable.
    disease : str, optional
        Human-readable disease name, used in the explanation.

    Returns
    -------
    dict
        ``available``, ``score`` (0-100), ``label``, ``level``,
        ``affected_area_percentage``, ``affected_regions``,
        ``largest_region_fraction``, ``confidence`` (trust in this estimate)
        and ``explanation``. When severity cannot be assessed, ``available``
        is False and only ``explanation`` is meaningful.
    """
    area = estimate_affected_area(pil_image)

    if not area.get("available"):
        return {
            "available": False,
            "explanation": area.get("explanation", "Affected area could not be estimated."),
        }

    affected_pct = float(area["affected_area_percentage"])

    # --- Score ---------------------------------------------------------
    # For a diseased leaf the score is simply "how much of the leaf is
    # visibly affected". For a healthy classification the measured area is
    # discounted because healthy leaves naturally contain some non-green
    # pixels that color segmentation cannot distinguish from disease.
    if status == "Healthy":
        raw_score = affected_pct * HEALTHY_SEVERITY_DISCOUNT
    else:
        raw_score = affected_pct

    score = int(round(max(0.0, min(100.0, raw_score))))
    label, level = severity_label_for_score(score)

    # --- Confidence in the estimate --------------------------------------
    # Higher when: the diagnosis itself is confident, classes are well
    # separated (margin), the image is mostly leaf (foliage coverage), and
    # symptoms were actually measurable.
    coverage_factor = min(1.0, area["foliage_ratio"] / FOLIAGE_FULL_COVERAGE)
    measured_factor = 1.0 if affected_pct >= MIN_MEASURABLE_AREA_PCT else 0.7
    margin = float(margin or 0.0)  # defensively handle None/0 from callers
    margin_factor = 0.85 + 0.15 * min(1.0, max(0.0, margin) / 0.20)
    severity_confidence = round(
        float(confidence) * (0.5 + 0.5 * coverage_factor) * measured_factor * margin_factor,
        4,
    )

    measured_note = (
        f"Severity estimate confidence: {severity_confidence:.2f}."
        if severity_confidence < 0.6
        else ""
    )

    return {
        "available": True,
        "score": score,
        "label": label,
        "level": level,
        "affected_area_percentage": round(affected_pct, 1),
        "affected_regions": area["affected_regions"],
        "largest_region_fraction": area["largest_region_fraction"],
        "confidence": severity_confidence,
        "explanation": _build_explanation(
            status, affected_pct, area["affected_regions"], disease, measured_note
        ),
    }
