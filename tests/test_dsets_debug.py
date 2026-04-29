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
