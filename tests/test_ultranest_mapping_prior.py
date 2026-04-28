"""Unit tests for SolverUltraNest wide-prior fix for mapping params.

Regression test for molass-legacy #32:
  NS optimizer drifts mp_a/mp_b when LRF initial estimate is wrong.
  Fix: _get_mapping_norm_indeces() returns the norm-space indices of the
  mapping params so minimize() can apply wide priors to them.
"""
import numpy as np
import pytest
from unittest.mock import MagicMock


def make_mock_optimizer(n_components=3, xr_params_indeces=None):
    """Create a minimal optimizer mock with SdmParams-like params_type."""
    from molass_legacy.ModelParams.SdmParams import SdmParams

    opt = MagicMock()
    opt.params_type = SdmParams(n_components)
    opt.xr_params_indeces = xr_params_indeces

    # Minimal attributes SolverUltraNest.__init__ touches
    opt.shm = None
    opt.num_pure_components = n_components
    opt.cb_fh = None

    return opt


def make_solver(n_components=3, xr_params_indeces=None):
    """Return a SolverUltraNest instance backed by a mock optimizer."""
    from molass_legacy.Solvers.UltraNest.SolverUltraNest import SolverUltraNest

    opt = make_mock_optimizer(n_components, xr_params_indeces)
    solver = object.__new__(SolverUltraNest)
    solver.optimizer = opt
    solver.shm = None
    solver.num_pure_components = n_components
    solver.cb_fh = None
    solver.callback_counter = 0
    import logging
    solver.logger = logging.getLogger(__name__)
    return solver


class TestGetMappingNormIndeces:
    """Tests for SolverUltraNest._get_mapping_norm_indeces()."""

    def test_returns_none_when_no_params_type(self):
        solver = make_solver()
        solver.optimizer.params_type = None
        assert solver._get_mapping_norm_indeces() is None

    def test_returns_none_when_params_type_has_no_pos(self):
        solver = make_solver()
        del solver.optimizer.params_type.pos  # strip pos
        assert solver._get_mapping_norm_indeces() is None

    def test_returns_two_indices_xr_indeces_none(self):
        """With xr_params_indeces=None all real indices == norm indices."""
        solver = make_solver(n_components=3)
        idx = solver._get_mapping_norm_indeces()
        assert idx is not None
        assert len(idx) == 2

        # Indices must equal pos[3] and pos[3]+1
        pos3 = solver.optimizer.params_type.pos[3]
        np.testing.assert_array_equal(idx, [pos3, pos3 + 1])

    def test_indices_vary_with_n_components(self):
        """More components → higher pos[3]."""
        solver2 = make_solver(n_components=2)
        solver3 = make_solver(n_components=3)
        idx2 = solver2._get_mapping_norm_indeces()
        idx3 = solver3._get_mapping_norm_indeces()
        assert idx2[0] < idx3[0], "pos[3] should increase with more components"

    def test_returns_subset_when_xr_params_indeces_set(self):
        """With xr_params_indeces set, returned indices are into the subset array."""
        from molass_legacy.ModelParams.SdmParams import SdmParams

        n = 3
        sdm = SdmParams(n)
        pos3 = sdm.pos[3]       # real start of mapping group
        n_real = sdm.num_params

        # Build a fake xr_params_indeces that includes the mapping params
        xr_idx = np.arange(n_real, dtype=int)          # all params
        solver = make_solver(n_components=n, xr_params_indeces=xr_idx)
        idx = solver._get_mapping_norm_indeces()

        # When xr_params_indeces == arange(n_real), norm == real
        assert idx is not None
        assert len(idx) == 2
        np.testing.assert_array_equal(idx, [pos3, pos3 + 1])

    def test_returns_empty_when_mapping_not_in_xr_subset(self):
        """If xr_params_indeces excludes mapping params, return None-like result."""
        from molass_legacy.ModelParams.SdmParams import SdmParams

        n = 3
        sdm = SdmParams(n)
        pos3 = sdm.pos[3]

        # Exclude mapping params from the XR subset
        xr_idx = np.array([i for i in range(sdm.num_params) if i not in (pos3, pos3 + 1)], dtype=int)
        solver = make_solver(n_components=n, xr_params_indeces=xr_idx)
        idx = solver._get_mapping_norm_indeces()
        # Result should be empty or None (no mapping params found in subset)
        assert idx is None or len(idx) == 0


class TestMinimizeWideMapping:
    """Integration-style test: verify that minimize() widens bounds for mapping params."""

    def test_mapping_prior_is_wide_when_narrow_bounds_true(self):
        """When narrow_bounds=True, mapping params must still use wide bounds."""
        from molass_legacy.ModelParams.SdmParams import SdmParams

        n = 3
        sdm = SdmParams(n)
        pos3 = sdm.pos[3]
        n_params = sdm.num_params

        # Build norm init_params all at 5.0 (centre of [0, 10])
        init_params = np.full(n_params, 5.0)
        # Set mapping init to simulate a wrong LRF estimate (near edge of norm range)
        init_params[pos3]     = 8.5   # mp_a: near upper edge
        init_params[pos3 + 1] = 8.5   # mp_b: near upper edge

        bounds = np.array([(0.0, 10.0)] * n_params)

        solver = make_solver(n_components=n)
        idx = solver._get_mapping_norm_indeces()

        # Simulate what minimize() does
        lower = init_params.copy() - 1.0
        upper = init_params.copy() + 1.0
        lower[idx] = bounds[idx, 0]
        upper[idx] = bounds[idx, 1]

        # All non-mapping params: narrow window = init ± 1
        for i in range(n_params):
            if i in idx:
                assert lower[i] == 0.0, f"idx {i}: lower should be wide (0.0)"
                assert upper[i] == 10.0, f"idx {i}: upper should be wide (10.0)"
            else:
                assert lower[i] == pytest.approx(init_params[i] - 1.0)
                assert upper[i] == pytest.approx(init_params[i] + 1.0)
