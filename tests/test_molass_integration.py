"""
Tests for Optimizer.MolassIntegration — molass-legacy#45.

Unit tests verify:
1. _get_slices_from_serial_settings() extracts correct (islice, jslice) tuples
2. try_patch_from_molass() patches DataSet arrays in-place on success
3. try_patch_from_molass() falls back silently when molass-library is absent
4. try_patch_from_molass() skips patch on shape mismatch (no exception)

Tests mock _load_corrected and SerialSettings to avoid needing real data.
"""
import unittest
import numpy as np
from unittest.mock import patch, MagicMock


PATCH_BASE = 'molass_legacy.Optimizer.MolassIntegration'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_trimming_info(start, stop, size):
    """Minimal TrimmingInfo-like mock with .get_slice()."""
    ti = MagicMock()
    ti.get_slice.return_value = slice(start, stop)
    return ti


def _make_dataset(frames, q, wl):
    """Minimal mock DataSet with correctly shaped numpy arrays."""
    ds = MagicMock()
    ds.xr_array = np.zeros((frames, q, 3))
    ds.xr_array[:, :, 1] = np.ones((frames, q)) * 2.0   # D (raw legacy)
    ds.xr_array[:, :, 2] = np.ones((frames, q)) * 0.1   # E (raw legacy)
    ds.uv_array = np.ones((wl, frames)) * 0.5            # U (raw legacy)
    return ds


def _make_molass_corrected(frames, q, wl, xr_E=True):
    """Mock corrected SSD returned by molass-library's corrected_copy()."""
    xr = MagicMock()
    xr.M = np.full((q, frames), 9.0)
    xr.E = np.full((q, frames), 0.02) if xr_E else None
    uv = MagicMock()
    uv.M = np.full((wl, frames), 7.0)
    corrected = MagicMock()
    corrected.xr = xr
    corrected.uv = uv
    return corrected


# ---------------------------------------------------------------------------
# Tests: _get_slices_from_serial_settings
# ---------------------------------------------------------------------------

class TestGetSlicesFromSerialSettings(unittest.TestCase):

    def _fn(self):
        from molass_legacy.Optimizer.MolassIntegration import _get_slices_from_serial_settings
        return _get_slices_from_serial_settings

    def test_extracts_correct_order(self):
        # restrict_list = [j(frame), i(spectral)]  → slices = (i-slice, j-slice)
        xr_j = _make_trimming_info(550, 1310, 1500)
        xr_i = _make_trimming_info(24, 990, 1026)
        uv_j = _make_trimming_info(464, 1403, 1500)
        uv_i = _make_trimming_info(101, 501, 501)
        fake = {
            'xr_restrict_list': [xr_j, xr_i],
            'uv_restrict_list': [uv_j, uv_i],
        }
        with patch('molass_legacy._MOLASS.SerialSettings.get_setting',
                   side_effect=lambda k: fake.get(k)):
            xr_slices, uv_slices = self._fn()()

        self.assertEqual(xr_slices, (slice(24, 990), slice(550, 1310)))
        self.assertEqual(uv_slices, (slice(101, 501), slice(464, 1403)))

    def test_returns_none_when_absent(self):
        with patch('molass_legacy._MOLASS.SerialSettings.get_setting',
                   return_value=None):
            xr_slices, uv_slices = self._fn()()
        self.assertIsNone(xr_slices)
        self.assertIsNone(uv_slices)


# ---------------------------------------------------------------------------
# Tests: try_patch_from_molass
# ---------------------------------------------------------------------------

class TestTryPatchFromMolassSuccess(unittest.TestCase):

    def _run(self, sd, corrected_sd, corrected_mock, frames, q, wl):
        from molass_legacy.Optimizer.MolassIntegration import try_patch_from_molass
        xr_slices = (slice(24, 990), slice(550, 1310))
        uv_slices = (slice(101, 501), slice(464, 1224))
        FakeSSD = MagicMock()
        with patch(PATCH_BASE + '._get_slices_from_serial_settings',
                   return_value=(xr_slices, uv_slices)), \
             patch(PATCH_BASE + '._load_corrected',
                   return_value=corrected_mock), \
             patch.dict('sys.modules', {
                 'molass': MagicMock(),
                 'molass.DataObjects': MagicMock(SecSaxsData=FakeSSD)}):
            try_patch_from_molass('fake_folder', sd, corrected_sd)

    def test_d_patched(self):
        frames, q, wl = 760, 966, 400
        sd = _make_dataset(frames, q, wl)
        corrected_sd = _make_dataset(frames, q, wl)
        self._run(sd, corrected_sd, _make_molass_corrected(frames, q, wl), frames, q, wl)
        np.testing.assert_array_equal(corrected_sd.xr_array[:, :, 1], 9.0)

    def test_e_patched(self):
        frames, q, wl = 760, 966, 400
        sd = _make_dataset(frames, q, wl)
        corrected_sd = _make_dataset(frames, q, wl)
        self._run(sd, corrected_sd, _make_molass_corrected(frames, q, wl), frames, q, wl)
        np.testing.assert_array_equal(corrected_sd.xr_array[:, :, 2], 0.02)

    def test_u_patched(self):
        frames, q, wl = 760, 966, 400
        sd = _make_dataset(frames, q, wl)
        corrected_sd = _make_dataset(frames, q, wl)
        self._run(sd, corrected_sd, _make_molass_corrected(frames, q, wl), frames, q, wl)
        np.testing.assert_array_equal(sd.uv_array, 7.0)

    def test_e_none_leaves_e_unchanged(self):
        frames, q, wl = 760, 966, 400
        sd = _make_dataset(frames, q, wl)
        corrected_sd = _make_dataset(frames, q, wl)
        E_before = corrected_sd.xr_array[:, :, 2].copy()
        self._run(sd, corrected_sd, _make_molass_corrected(frames, q, wl, xr_E=False),
                  frames, q, wl)
        np.testing.assert_array_equal(corrected_sd.xr_array[:, :, 2], E_before)
        # D still patched
        np.testing.assert_array_equal(corrected_sd.xr_array[:, :, 1], 9.0)


class TestTryPatchFromMolassFallback(unittest.TestCase):

    def test_no_patch_on_shape_mismatch(self):
        frames, q, wl = 760, 966, 400
        sd = _make_dataset(frames, q, wl)
        corrected_sd = _make_dataset(frames, q, wl)
        D_before = corrected_sd.xr_array[:, :, 1].copy()

        wrong = _make_molass_corrected(frames + 10, q, wl)  # wrong frame count
        xr_slices = (slice(24, 990), slice(550, 1310))
        uv_slices = (slice(101, 501), slice(464, 1224))
        FakeSSD = MagicMock()

        from molass_legacy.Optimizer.MolassIntegration import try_patch_from_molass
        with patch(PATCH_BASE + '._get_slices_from_serial_settings',
                   return_value=(xr_slices, uv_slices)), \
             patch(PATCH_BASE + '._load_corrected', return_value=wrong), \
             patch.dict('sys.modules', {
                 'molass': MagicMock(),
                 'molass.DataObjects': MagicMock(SecSaxsData=FakeSSD)}):
            try_patch_from_molass('fake_folder', sd, corrected_sd)

        np.testing.assert_array_equal(corrected_sd.xr_array[:, :, 1], D_before)

    def test_no_exception_when_slices_none(self):
        frames, q, wl = 760, 966, 400
        sd = _make_dataset(frames, q, wl)
        corrected_sd = _make_dataset(frames, q, wl)
        D_before = corrected_sd.xr_array[:, :, 1].copy()
        FakeSSD = MagicMock()

        from molass_legacy.Optimizer.MolassIntegration import try_patch_from_molass
        with patch(PATCH_BASE + '._get_slices_from_serial_settings',
                   return_value=(None, None)), \
             patch.dict('sys.modules', {
                 'molass': MagicMock(),
                 'molass.DataObjects': MagicMock(SecSaxsData=FakeSSD)}):
            try_patch_from_molass('fake_folder', sd, corrected_sd)  # must not raise

        np.testing.assert_array_equal(corrected_sd.xr_array[:, :, 1], D_before)

    def test_no_exception_when_load_corrected_raises(self):
        frames, q, wl = 760, 966, 400
        sd = _make_dataset(frames, q, wl)
        corrected_sd = _make_dataset(frames, q, wl)
        D_before = corrected_sd.xr_array[:, :, 1].copy()

        xr_slices = (slice(24, 990), slice(550, 1310))
        uv_slices = (slice(101, 501), slice(464, 1224))
        FakeSSD = MagicMock()

        from molass_legacy.Optimizer.MolassIntegration import try_patch_from_molass
        with patch(PATCH_BASE + '._get_slices_from_serial_settings',
                   return_value=(xr_slices, uv_slices)), \
             patch(PATCH_BASE + '._load_corrected',
                   side_effect=RuntimeError("disk read failed")), \
             patch.dict('sys.modules', {
                 'molass': MagicMock(),
                 'molass.DataObjects': MagicMock(SecSaxsData=FakeSSD)}):
            try_patch_from_molass('fake_folder', sd, corrected_sd)  # must not raise

        np.testing.assert_array_equal(corrected_sd.xr_array[:, :, 1], D_before)


if __name__ == '__main__':
    unittest.main()
