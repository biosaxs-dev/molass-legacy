"""
Tests for Optimizer.DsetsDebug — molass-legacy#34.

These are unit tests that do NOT require real data on disk.  They mock out
the heavy dependencies (FullOptInput, DataTreatment) so the tests run fast
and without an actual SEC-SAXS data folder.
"""
import os
import unittest
import tempfile
import numpy as np
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# Helpers to build minimal mock dsets structures
# ---------------------------------------------------------------------------

def _make_curve(x_start, n, y_scale=1.0):
    """Return a minimal mock elution curve with .x and .y arrays."""
    c = MagicMock()
    c.x = np.arange(x_start, x_start + n, dtype=float)
    c.y = np.exp(-0.5 * ((c.x - (x_start + n / 2)) / (n / 6)) ** 2) * y_scale
    return c


def _make_opt_dsets(xr_start=100, uv_start=50, n_xr=200, n_uv=300,
                    n_q=50, n_wl=40):
    """Return a mock OptDataSets with minimal shape / value data."""
    xr_curve = _make_curve(xr_start, n_xr)
    uv_curve = _make_curve(uv_start, n_uv, y_scale=2.0)
    D = np.random.default_rng(42).uniform(0.01, 1.0, size=(n_q, n_xr))
    U = np.random.default_rng(7).uniform(0.0, 0.5, size=(n_wl, n_uv))
    rg_curve = MagicMock()

    mock = MagicMock()
    mock.dsets = ((xr_curve, D), rg_curve, (uv_curve, U))
    return mock


# ---------------------------------------------------------------------------
# Test compare_dsets
# ---------------------------------------------------------------------------

class TestCompareDsets(unittest.TestCase):
    def _import(self):
        from molass_legacy.Optimizer.DsetsDebug import compare_dsets
        return compare_dsets

    def test_identical_dsets_all_diffs_zero(self):
        compare_dsets = self._import()
        ds = _make_opt_dsets()
        summary = compare_dsets(ds, ds, label_a="A", label_b="A-copy")

        self.assertEqual(summary["xr_x_maxdiff"], 0.0)
        self.assertEqual(summary["uv_x_maxdiff"], 0.0)
        self.assertEqual(summary["D_maxdiff"], 0.0)
        self.assertEqual(summary["U_maxdiff"], 0.0)

    def test_shape_mismatch_returns_none(self):
        compare_dsets = self._import()
        ds_a = _make_opt_dsets(n_xr=200)
        ds_b = _make_opt_dsets(n_xr=180)
        summary = compare_dsets(ds_a, ds_b)

        self.assertIsNone(summary["xr_x_maxdiff"])
        self.assertIsNone(summary["D_maxdiff"])

    def test_offset_x_detected(self):
        compare_dsets = self._import()
        ds_a = _make_opt_dsets(xr_start=100)
        ds_b = _make_opt_dsets(xr_start=162)  # 62-frame offset
        summary = compare_dsets(ds_a, ds_b)

        self.assertIsNotNone(summary["xr_x_maxdiff"])
        self.assertAlmostEqual(summary["xr_x_maxdiff"], 62.0, places=6)

    def test_summary_keys_present(self):
        compare_dsets = self._import()
        ds = _make_opt_dsets()
        summary = compare_dsets(ds, ds)

        for key in ("xr_x_a", "xr_x_b", "uv_x_a", "uv_x_b",
                    "D_shape_a", "D_shape_b", "U_shape_a", "U_shape_b",
                    "xr_x_maxdiff", "uv_x_maxdiff", "D_maxdiff", "U_maxdiff"):
            self.assertIn(key, summary, f"Key '{key}' missing from summary")


# ---------------------------------------------------------------------------
# Test get_mp_b_index
# ---------------------------------------------------------------------------

class TestGetMpBIndex(unittest.TestCase):
    def test_returns_pos3_plus_one(self):
        from molass_legacy.Optimizer.DsetsDebug import get_mp_b_index
        optimizer = MagicMock()
        optimizer.params_type.pos = [0, 10, 20, 30, 40]
        self.assertEqual(get_mp_b_index(optimizer), 31)

    def test_raises_when_no_params_type(self):
        from molass_legacy.Optimizer.DsetsDebug import get_mp_b_index
        optimizer = MagicMock(spec=[])  # no params_type attribute
        with self.assertRaises(ValueError):
            get_mp_b_index(optimizer)


# ---------------------------------------------------------------------------
# Test sweep_mp_b
# ---------------------------------------------------------------------------

class TestSweepMpB(unittest.TestCase):
    def test_returns_arrays_of_correct_length(self):
        from molass_legacy.Optimizer.DsetsDebug import sweep_mp_b

        # Toy objective: fv = (mp_b - 5)^2 so minimum at mp_b=5
        optimizer = MagicMock()
        optimizer.objective_func.side_effect = lambda p: (p[2] - 5.0) ** 2
        init_params = np.zeros(5)

        mp_b_values, fv_values = sweep_mp_b(
            optimizer, init_params, mp_b_range=(-10.0, 20.0), n_points=31, mp_b_index=2
        )

        self.assertEqual(len(mp_b_values), 31)
        self.assertEqual(len(fv_values), 31)

    def test_minimum_at_correct_mp_b(self):
        from molass_legacy.Optimizer.DsetsDebug import sweep_mp_b

        optimizer = MagicMock()
        optimizer.objective_func.side_effect = lambda p: (p[2] - 7.5) ** 2
        init_params = np.zeros(5)

        mp_b_values, fv_values = sweep_mp_b(
            optimizer, init_params, mp_b_range=(-20.0, 20.0), n_points=401, mp_b_index=2
        )

        best_mp_b = mp_b_values[np.argmin(fv_values)]
        self.assertAlmostEqual(best_mp_b, 7.5, delta=0.2)

    def test_other_params_unchanged(self):
        from molass_legacy.Optimizer.DsetsDebug import sweep_mp_b

        received = []

        def _obj(p):
            received.append(p.copy())
            return 0.0

        optimizer = MagicMock()
        optimizer.objective_func.side_effect = _obj

        init_params = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        sweep_mp_b(optimizer, init_params, mp_b_range=(0.0, 1.0), n_points=3, mp_b_index=2)

        for p in received:
            np.testing.assert_array_equal(p[[0, 1, 3, 4]], init_params[[0, 1, 3, 4]])


# ---------------------------------------------------------------------------
# Test get_dsets_impl ElCurve override (molass-legacy#38)
# ---------------------------------------------------------------------------

class TestGetDsetsImplElCurveOverride(unittest.TestCase):
    """Unit tests for the molass-legacy#38 fix.

    Verify that get_dsets_impl replaces xr_curve.y and uv_curve.y with the
    parent's EGH-fitted values when ip_xr_elcurve_y.npy / ip_uv_elcurve_y.npy
    exist in the optimizer_folder.
    """

    def _run_with_override(self, tmpdir, override_xr=True, override_uv=True,
                            override_D=False, override_U=False, override_E=False):
        """Run get_dsets_impl with mocked sd and rg_folder, return dsets tuple."""
        from molass_legacy.Optimizer.OptDataSets import get_dsets_impl
        from scipy.interpolate import InterpolatedUnivariateSpline

        n_xr, n_uv = 200, 300
        xr_x = np.arange(500, 500 + n_xr, dtype=float)
        uv_x = np.arange(400, 400 + n_uv, dtype=float)

        # "legacy-smoothed" curves returned by sd (wrong values)
        xr_y_legacy = np.ones(n_xr) * 0.5
        uv_y_legacy = np.ones(n_uv) * 0.8

        # Parent's EGH-fitted curves (correct values, Gaussian-shaped)
        xr_y_parent = np.exp(-0.5 * ((xr_x - 600) / 30) ** 2)
        uv_y_parent = np.exp(-0.5 * ((uv_x - 550) / 40) ** 2)

        # Build mock xr_curve and uv_curve
        xr_curve = MagicMock()
        xr_curve.x = xr_x
        xr_curve.y = xr_y_legacy.copy()

        uv_curve = MagicMock()
        uv_curve.x = uv_x
        uv_curve.y = uv_y_legacy.copy()
        # Spline built with 0-based domain (triggers molass-legacy#34 branch)
        uv_curve.spline = InterpolatedUnivariateSpline(np.arange(n_uv, dtype=float), uv_y_legacy, ext=3)
        uv_curve.sy = uv_y_legacy  # legacy has sy attribute

        D = np.ones((50, n_xr))
        U = np.ones((40, n_uv))
        D_parent = np.random.default_rng(99).uniform(0.01, 1.0, size=(50, n_xr))
        U_parent = np.random.default_rng(77).uniform(0.0, 0.5, size=(40, n_uv))
        E_legacy = np.ones((50, n_xr)) * 0.01   # sd.intensity_array (legacy)
        E_parent = np.random.default_rng(55).uniform(0.001, 0.05, size=(50, n_xr))

        # sd returns the legacy curves; intensity_array[:,:,2].T → E_legacy
        # intensity_array shape is (n_frames, n_q, 3), so [:,:,2].T gives (n_q, n_frames) = E_legacy
        sd = MagicMock()
        sd.get_xr_data_separate_ly.return_value = (D, E_legacy, np.linspace(0.01, 0.3, 50), xr_curve)
        sd.intensity_array = np.zeros((n_xr, 50, 3))   # (n_frames, n_q, 3)
        sd.intensity_array[:, :, 2] = E_legacy.T        # E_legacy.T is (n_frames, n_q)
        sd.get_uv_data_separate_ly.return_value = (U, None, np.linspace(200, 600, 40), uv_curve)

        # rg-curve setup: write trust.txt and required files so rg branch is satisfied
        rg_folder = os.path.join(tmpdir, "rg-curve")
        os.makedirs(rg_folder)
        open(os.path.join(rg_folder, "trust.txt"), "w").close()
        for fname in ["segments.txt", "qualities.txt", "slices.txt", "states.txt", "baseline_type.txt"]:
            open(os.path.join(rg_folder, fname), "w").close()
        open(os.path.join(rg_folder, "ok.stamp"), "w").close()

        # RgCurveProxy mock
        with patch("molass_legacy.Optimizer.OptDataSets.get_setting") as mock_get, \
             patch("molass_legacy.Optimizer.OptDataSets.set_setting"), \
             patch("molass_legacy.RgProcess.RgCurve.check_rg_folder", return_value=True), \
             patch("molass_legacy.RgProcess.RgCurveProxy.RgCurveProxy") as mock_proxy:
            mock_get.return_value = True  # trust_rg_curve_folder=True
            mock_proxy.return_value = MagicMock()

            # Write parent npy files to optimizer_folder
            if override_xr:
                np.save(os.path.join(tmpdir, "ip_xr_elcurve_y.npy"), xr_y_parent)
            if override_uv:
                np.save(os.path.join(tmpdir, "ip_uv_elcurve_y.npy"), uv_y_parent)
            if override_D:
                np.save(os.path.join(tmpdir, "ip_xr_D.npy"), D_parent)
            if override_U:
                np.save(os.path.join(tmpdir, "ip_uv_U.npy"), U_parent)
            if override_E:
                np.save(os.path.join(tmpdir, "ip_xr_E.npy"), E_parent)

            result = get_dsets_impl(sd, MagicMock(), rg_folder=rg_folder)

        return result, xr_y_parent, uv_y_parent, xr_y_legacy, uv_y_legacy, D_parent, U_parent, D, U, E_parent, E_legacy

    def test_xr_curve_y_overridden_when_npy_exists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result, xr_y_parent, _, _, _, _, _, _, _, _, _ = self._run_with_override(tmpdir, override_xr=True, override_uv=False)
            np.testing.assert_array_almost_equal(result[0][0].y, xr_y_parent)

    def test_xr_curve_y_unchanged_when_no_npy(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result, _, _, xr_y_legacy, _, _, _, _, _, _, _ = self._run_with_override(tmpdir, override_xr=False, override_uv=False)
            np.testing.assert_array_almost_equal(result[0][0].y, xr_y_legacy)

    def test_uv_curve_y_overridden_when_npy_exists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result, _, uv_y_parent, _, _, _, _, _, _, _, _ = self._run_with_override(tmpdir, override_xr=False, override_uv=True)
            np.testing.assert_array_almost_equal(result[2][0].y, uv_y_parent)

    def test_uv_curve_spline_rebuilt_with_new_y(self):
        """After override, uv_curve.spline must interpolate the new (parent) y values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result, _, uv_y_parent, _, uv_y_legacy, _, _, _, _, _, _ = self._run_with_override(tmpdir, override_xr=False, override_uv=True)
            uv_curve = result[2][0]
            uv_x = uv_curve.x
            # Spline should match new y at interior points, not old y
            spline_vals = uv_curve.spline(uv_x[10:-10])
            np.testing.assert_array_almost_equal(spline_vals, uv_y_parent[10:-10], decimal=5)

    def test_xr_D_overridden_when_npy_exists(self):
        """D matrix must be replaced with parent's corrected matrix (molass-legacy#39)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result, _, _, _, _, D_parent, _, D_legacy, _, _, _ = self._run_with_override(
                tmpdir, override_xr=False, override_uv=False, override_D=True, override_U=False)
            np.testing.assert_array_equal(result[0][1], D_parent)
            # Confirm D_legacy and D_parent are genuinely different
            self.assertFalse(np.allclose(D_legacy, D_parent))

    def test_uv_U_overridden_when_npy_exists(self):
        """U matrix must be replaced with parent's corrected matrix (molass-legacy#39)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result, _, _, _, _, _, U_parent, _, U_legacy, _, _ = self._run_with_override(
                tmpdir, override_xr=False, override_uv=False, override_D=False, override_U=True)
            np.testing.assert_array_equal(result[2][1], U_parent)
            # Confirm U_legacy and U_parent are genuinely different
            self.assertFalse(np.allclose(U_legacy, U_parent))

    def test_xr_E_returned_as_override_when_npy_exists(self):
        """E matrix override must be returned via return_e_override=True (molass-legacy#39)."""
        from molass_legacy.Optimizer.OptDataSets import get_dsets_impl
        from scipy.interpolate import InterpolatedUnivariateSpline
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            result, _, _, _, _, _, _, _, _, E_parent, E_legacy = self._run_with_override(
                tmpdir, override_xr=False, override_uv=False, override_E=True)
            # get_dsets_impl without return_e_override returns dsets only — test via OptDataSets
            # Re-run with return_e_override=True to verify E_override is returned
            n_xr, n_uv = 200, 300
            xr_x = np.arange(500, 500 + n_xr, dtype=float)
            uv_x = np.arange(400, 400 + n_uv, dtype=float)
            xr_curve2 = MagicMock(); xr_curve2.x = xr_x; xr_curve2.y = np.ones(n_xr)
            uv_curve2 = MagicMock(); uv_curve2.x = uv_x; uv_curve2.y = np.ones(n_uv)
            uv_curve2.spline = InterpolatedUnivariateSpline(np.arange(n_uv, dtype=float), np.ones(n_uv), ext=3)
            uv_curve2.sy = np.ones(n_uv)
            D2 = np.ones((50, n_xr))
            U2 = np.ones((40, n_uv))
            sd2 = MagicMock()
            sd2.get_xr_data_separate_ly.return_value = (D2, np.ones((50, n_xr)) * 0.01, np.linspace(0.01, 0.3, 50), xr_curve2)
            sd2.intensity_array = np.zeros((n_xr, 50, 3)); sd2.intensity_array[:, :, 2] = (np.ones((50, n_xr)) * 0.01).T
            sd2.get_uv_data_separate_ly.return_value = (U2, None, np.linspace(200, 600, 40), uv_curve2)
            rg_folder2 = os.path.join(tmpdir, "rg-curve2")
            os.makedirs(rg_folder2)
            for fn in ["trust.txt", "segments.txt", "qualities.txt", "slices.txt", "states.txt", "baseline_type.txt", "ok.stamp"]:
                open(os.path.join(rg_folder2, fn), "w").close()
            np.save(os.path.join(tmpdir, "ip_xr_E.npy"), E_parent)
            with patch("molass_legacy.Optimizer.OptDataSets.get_setting") as mock_get, \
                 patch("molass_legacy.Optimizer.OptDataSets.set_setting"), \
                 patch("molass_legacy.RgProcess.RgCurve.check_rg_folder", return_value=True), \
                 patch("molass_legacy.RgProcess.RgCurveProxy.RgCurveProxy") as mock_proxy:
                mock_get.return_value = True
                mock_proxy.return_value = MagicMock()
                _, E_override = get_dsets_impl(sd2, MagicMock(), rg_folder=rg_folder2, return_e_override=True)
            np.testing.assert_array_equal(E_override, E_parent)
            self.assertFalse(np.allclose(E_legacy, E_parent))


# ---------------------------------------------------------------------------
# Test reconstruct_subprocess_dsets (mocked)
# ---------------------------------------------------------------------------

class TestReconstructSubprocessDsets(unittest.TestCase):
    def test_raises_when_no_trimming_txt(self):
        from molass_legacy.Optimizer.DsetsDebug import reconstruct_subprocess_dsets
        with tempfile.TemporaryDirectory() as tmpdir:
            # No trimming.txt in tmpdir — must raise FileNotFoundError
            with self.assertRaises(FileNotFoundError):
                reconstruct_subprocess_dsets(tmpdir)

    def test_reads_in_folder_from_file(self):
        """When in_folder.txt is present, it is used without consulting settings."""
        from molass_legacy.Optimizer.DsetsDebug import _get_in_folder_from_work_folder
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "in_folder.txt")
            with open(path, "w") as fh:
                fh.write("/data/sample1\n")
            self.assertEqual(_get_in_folder_from_work_folder(tmpdir), "/data/sample1")

    def test_returns_none_when_no_in_folder_file(self):
        from molass_legacy.Optimizer.DsetsDebug import _get_in_folder_from_work_folder
        with tempfile.TemporaryDirectory() as tmpdir:
            self.assertIsNone(_get_in_folder_from_work_folder(tmpdir))

    def test_optimizer_folder_derived_correctly(self):
        """work_folder = optimizer_folder/jobs/000 → optimizer_folder two levels up."""
        from molass_legacy.Optimizer.DsetsDebug import _derive_optimizer_folder
        work = os.path.join("C:", "analysis", "optimized", "jobs", "000")
        expected = os.path.join("C:", "analysis", "optimized")
        self.assertEqual(_derive_optimizer_folder(work), expected)


# ---------------------------------------------------------------------------
# Test that BackRunner and InProcessRunner write in_folder.txt
# ---------------------------------------------------------------------------

class TestInFolderFileSaved(unittest.TestCase):
    """Smoke-test that the sentinel file is written to the work folder."""

    def _check_in_folder_written(self, runner_func, work_folder, in_folder):
        """Call runner_func and assert in_folder.txt ends up in work_folder."""
        expected_path = os.path.join(work_folder, "in_folder.txt")
        runner_func()
        self.assertTrue(
            os.path.exists(expected_path),
            f"in_folder.txt not written to {work_folder}"
        )
        with open(expected_path) as fh:
            content = fh.read().strip()
        self.assertEqual(content, in_folder)


if __name__ == "__main__":
    unittest.main()
