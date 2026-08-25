"""
Test that MplMonitor's max_trials auto-resume / "Resume Job" button reseed
from the GLOBAL best across all completed jobs, not just the just-completed
trial's own state.

See: https://github.com/biosaxs-dev/molass-library/issues/257
"""
import os
import tempfile
import logging
import numpy as np
from unittest.mock import MagicMock


def _write_callback_txt(folder, entries):
    """entries: list of (counter, fv, x_array)"""
    os.makedirs(folder, exist_ok=True)
    cb_path = os.path.join(folder, "callback.txt")
    with open(cb_path, "w") as f:
        for counter, fv, x in entries:
            f.write(f"t=2026-01-01 00:00:{counter:02d}\n")
            f.write("x=\n")
            f.write("[" + " ".join(str(v) for v in x) + "]\n")
            f.write(f"f={fv}\n")
            f.write("a=True\n")
            f.write(f"c={counter}\n")


def _make_mon(optimizer_folder, init_params):
    """Bare MplMonitor bypassing __init__ (needs filesystem + SerialSettings)."""
    from molass_legacy.Optimizer.MplMonitor import MplMonitor
    mon = object.__new__(MplMonitor)
    mon.optimizer_folder = optimizer_folder
    mon.init_params = init_params
    mon.logger = logging.getLogger("test_mplmonitor_global_best_reseed")
    return mon


def test_global_best_used_instead_of_just_completed_trial():
    """A worse-performing later trial must NOT override an earlier better one."""
    with tempfile.TemporaryDirectory() as tmpdir:
        optimizer_folder = os.path.join(tmpdir, "optimized")
        job_a = os.path.join(optimizer_folder, "jobs", "000")
        job_b = os.path.join(optimizer_folder, "jobs", "001")  # just completed, worse
        x_a = [1.1, 1.1, 1.1]  # better (lower fv)
        x_b = [2.2, 2.2, 2.2]  # worse (higher fv) -- the "just completed" trial
        _write_callback_txt(job_a, [(1, -2.0, x_a)])
        _write_callback_txt(job_b, [(1, -0.5, x_b)])

        init_params = np.zeros(3)
        mon = _make_mon(optimizer_folder, init_params)
        result = mon._get_global_best_params()

        np.testing.assert_array_almost_equal(result, x_a)


def test_falls_back_to_local_best_when_no_jobs_dir():
    """No jobs dir at all -> falls back to get_best_params() (local best)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        optimizer_folder = os.path.join(tmpdir, "optimized")  # jobs/ doesn't exist
        init_params = np.array([9.0, 9.0])
        mon = _make_mon(optimizer_folder, init_params)
        mon.get_best_params = MagicMock(return_value=init_params)

        result = mon._get_global_best_params()

        mon.get_best_params.assert_called_once()
        np.testing.assert_array_almost_equal(result, init_params)
