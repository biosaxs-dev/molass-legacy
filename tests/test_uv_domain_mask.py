"""
Tests for NumericalUtils.compute_uv_domain_mask, the frame mask that excludes
extrapolated UV frames from UV_2D_fitting/UV_LRF_residual in BasicOptimizer.

Reproduces the EcoCas3 domain-mismatch scenario (molass-library#270 follow-up,
molass-researcher/experiments/44_trimming_adjustment): XR frames span a wider
range than the really-measured UV frames, and uv_x (each XR frame mapped into
UV coordinates) extends past the real UV domain at both ends.
"""
import numpy as np
from molass_legacy.Optimizer.NumericalUtils import compute_uv_domain_mask


def test_excludes_frames_outside_real_uv_domain():
    uv_x = np.arange(0, 1599)          # full XR-mapped domain, as in EcoCas3
    lo, hi = 429, 1217                 # real measured UV domain
    mask = compute_uv_domain_mask(uv_x, lo, hi)
    assert mask.sum() == hi - lo + 1
    assert mask[lo] and mask[hi]       # inclusive boundaries
    assert not mask[lo - 1]
    assert not mask[hi + 1]


def test_all_valid_when_uv_x_fully_within_domain():
    uv_x = np.linspace(500, 900, 50)
    mask = compute_uv_domain_mask(uv_x, 0, 1599)
    assert mask.all()


def test_falls_back_to_all_true_when_too_few_valid_frames():
    # domain covers none of uv_x -- naive masking would leave 0 valid frames
    uv_x = np.arange(0, 100)
    mask = compute_uv_domain_mask(uv_x, 5000, 6000)
    assert mask.all()


def test_min_valid_threshold_is_configurable():
    uv_x = np.arange(0, 100)
    lo, hi = 40, 45   # exactly 6 frames valid
    mask_default = compute_uv_domain_mask(uv_x, lo, hi)              # min_valid=10 -> fallback
    mask_lenient = compute_uv_domain_mask(uv_x, lo, hi, min_valid=5)  # -> kept as-is
    assert mask_default.all()
    assert mask_lenient.sum() == 6


def test_boundary_equal_lo_hi_keeps_single_frame_but_falls_back():
    # a single matching frame is below the default min_valid -> fallback to all-True,
    # avoiding a near-degenerate (1-point) norm during optimization
    uv_x = np.arange(0, 50)
    mask = compute_uv_domain_mask(uv_x, 10, 10)
    assert mask.all()
