"""
molass-legacy#104: FlangeLimit.get_limit_impl raised UnboundLocalError ('is_limit') when no flange candidate was found,
which happens for a smooth stated error without the small-angle rise and wide-angle upturn of measured data
(for example simulated data with the photon-counting error of Sedlak et al. 2017). The limit must then be the corner
of the rightmost error-upturn, i.e. the minimum of the error when there is no upturn, and get_limit() must not log a failure.
"""
import logging

import numpy as np
import pytest

from molass_legacy.Trimming.FlangeLimit import FlangeLimit, LIMIT_TYPE_WA_CORNER


class _Curve:
    def __init__(self, i):
        self.i = i

    def get_primarypeak_i(self):
        return self.i


def _smooth_error_data(n_q=773, n_frames=220):
    q = np.linspace(0.00727, 0.4075, n_q)
    frames = np.arange(n_frames)
    elution = np.exp(-0.5*((frames - 83)/8.0)**2)
    data = 0.07*np.exp(-(q*28.0)**2/3)[:, None]*elution[None, :] + 1e-4
    error = np.repeat((2e-4/np.sqrt(q))[:, None], n_frames, axis=1)      # decreases with q, no upturn
    return q, data, error, _Curve(83)


def test_no_flange_candidate_gives_a_wide_angle_limit_and_no_exception(caplog):
    q, data, error, curve = _smooth_error_data()
    flange = FlangeLimit(data, error, curve, q)
    with caplog.at_level(logging.WARNING):
        limit = flange.get_limit()
    assert not any("get_limit_impl failed" in r.getMessage() for r in caplog.records)
    assert limit is not None and 0 < limit <= len(q)
    assert flange.limit_type == LIMIT_TYPE_WA_CORNER


def test_get_limit_impl_itself_does_not_raise():
    q, data, error, curve = _smooth_error_data()
    flange = FlangeLimit(data, error, curve, q)
    flange.get_limit_impl(curve, data, error, flange.logger)           # was UnboundLocalError
    assert flange.limit is not None
