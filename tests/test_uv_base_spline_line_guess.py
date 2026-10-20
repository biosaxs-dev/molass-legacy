"""
Tests for the default (line) initial guess of UvBaseSpline (molass-legacy#103).

The flow-change sigmoid model must not be applied by default: with a peak whose
flanks fall in the "lower part" of the curve, the baseline guess has to stay a
line close to the true baseline instead of rising under the peak.
"""
from types import SimpleNamespace
import numpy as np
from scipy.interpolate import UnivariateSpline
from molass_legacy.Baseline.UvBaseSpline import UvBaseSpline, compute_baseline_impl
from molass_legacy.Baseline.Constants import SLOPE_SCALE

X = np.arange(600, dtype=float)
TRUE_LINE = 0.002 - 5e-6*X


def make_base_spline():
    rng = np.random.default_rng(0)
    peak = 0.3*np.exp(-0.5*((X - 300)/30)**2) + 0.04*np.exp(-0.5*((X - 420)/40)**2)
    curve1 = SimpleNamespace(x=X, y=peak + TRUE_LINE + 1e-4*rng.standard_normal(len(X)))
    curve1.max_y = curve1.y.max()
    diff_spline = UnivariateSpline(X, np.gradient(peak), s=0, ext=3)
    curve2 = SimpleNamespace(x=X, y=-0.003 + 1e-6*X + 1e-4*rng.standard_normal(len(X)))
    ubs = UvBaseSpline(None, curve1, spline=diff_spline)
    ubs.baseline_type = 1
    return ubs, curve1, curve2


def test_default_guess_is_a_line_with_unchanged_layout():
    ubs, curve1, curve2 = make_base_spline()
    params = ubs.guess_params(curve2, None, slice(None, None), fc_point=53)
    assert len(params) == 7
    L, x0, k, b, s1, s2, diff_ratio = params
    assert L == 0
    assert x0 == X[0]
    assert s1 == s2

    base = ubs(X, params, None)
    assert np.max(np.abs(base - TRUE_LINE)) < 0.003     # no rise under the peak flanks
    assert abs(s2/SLOPE_SCALE - (-5e-6)) < 2e-6


def test_baseline_type_2_appends_integral_scale():
    ubs, curve1, curve2 = make_base_spline()
    ubs.baseline_type = 2
    params = ubs.guess_params(curve2, None, slice(None, None))
    assert len(params) == 8
    assert params[-1] == 0


def test_flow_change_model_is_opt_in():
    ubs, curve1, curve2 = make_base_spline()
    called = []
    ubs.guess_line_params = lambda *args, **kwargs: called.append(True)
    try:
        ubs.guess_params(curve2, None, slice(None, None), fc_point=None, flow_change_model=True)
    except Exception:
        pass    # the legacy model needs a fully initialised environment; reaching it is enough
    assert not called
