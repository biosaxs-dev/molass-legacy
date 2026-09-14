"""
Tests for SdmEstimator.get_colparam_bounds() N0 bound (molass-legacy#98).

N0 (mobile-phase plate number) is a column/flow-rate property, not sample-
dependent. get_colparam_bounds() previously hardcoded a static (1600, 60000)
range regardless of dataset -- 37x wide, and shown (via a real 3-component
SDM GUI run) to let N0 drift to very different values under DE (29112.8) vs
BH (9490.9). This anchors the bound on the column's rated plate count
(num_plates_pc, already in SerialSettings) instead, which comfortably
contained both of those converged results in the investigation that
motivated this fix.
"""
import pytest
from molass_legacy.Estimators.SdmEstimator import SdmEstimator


class _DummyEditor:
    """Minimal stand-in -- get_colparam_bounds() never touches sd/ecurves."""
    def __init__(self, num_components=3):
        self.sd = None
        self.corrected_sd = None
        self.ecurves = None
        self.n_components = num_components

    def get_n_components(self):
        return self.n_components


@pytest.fixture
def restore_num_plates_pc():
    from molass_legacy._MOLASS.SerialSettings import get_setting, set_setting
    original = get_setting("num_plates_pc")
    yield
    set_setting("num_plates_pc", original)


def test_n0_bound_anchored_on_num_plates_pc(restore_num_plates_pc):
    from molass_legacy._MOLASS.SerialSettings import set_setting
    set_setting("num_plates_pc", 14400)

    estimator = SdmEstimator(_DummyEditor(), pore_dist='mono')
    bounds = estimator.get_colparam_bounds()
    n0_lo, n0_hi = bounds[4]   # G1200 layout: [N, K, x0, poresize, N0, tI, k]

    assert (n0_lo, n0_hi) == pytest.approx((7200.0, 28800.0))


def test_n0_bound_scales_with_column_type(restore_num_plates_pc):
    from molass_legacy._MOLASS.SerialSettings import set_setting
    set_setting("num_plates_pc", 20000)

    estimator = SdmEstimator(_DummyEditor(), pore_dist='mono')
    n0_lo, n0_hi = estimator.get_colparam_bounds()[4]
    assert (n0_lo, n0_hi) == pytest.approx((10000.0, 40000.0))


def test_n0_bound_falls_back_to_static_default_when_setting_non_positive(restore_num_plates_pc):
    from molass_legacy._MOLASS.SerialSettings import set_setting
    set_setting("num_plates_pc", 0)

    estimator = SdmEstimator(_DummyEditor(), pore_dist='mono')
    n0_lo, n0_hi = estimator.get_colparam_bounds()[4]
    assert (n0_lo, n0_hi) == (1600, 60000)


def test_n0_bound_present_for_lognormal_variant(restore_num_plates_pc):
    from molass_legacy._MOLASS.SerialSettings import set_setting
    set_setting("num_plates_pc", 14400)

    estimator = SdmEstimator(_DummyEditor(), pore_dist='lognormal')
    bounds = estimator.get_colparam_bounds()
    # G1300 layout: [N, K, x0, mu, sigma, N0, tI, k]
    n0_lo, n0_hi = bounds[5]
    assert (n0_lo, n0_hi) == pytest.approx((7200.0, 28800.0))


def test_regression_analysis002_003_n0_within_bound(restore_num_plates_pc):
    """DE (analysis-003) converged N0=29112.8, BH (analysis-002) N0=9490.9,
    both with num_plates_pc=14400. The new bound should contain BH's result
    and come within ~1% of DE's (documented in the issue as an accepted,
    much closer miss than the old static/estimate-based bounds)."""
    from molass_legacy._MOLASS.SerialSettings import set_setting
    set_setting("num_plates_pc", 14400)

    estimator = SdmEstimator(_DummyEditor(), pore_dist='mono')
    n0_lo, n0_hi = estimator.get_colparam_bounds()[4]

    bh_n0 = 9490.883
    de_n0 = 29112.829
    assert n0_lo <= bh_n0 <= n0_hi
    assert abs(de_n0 - n0_hi) / de_n0 < 0.02
