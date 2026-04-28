"""
Tests for EghAdvansedParams.get_parameter_names() — issue #33.

Verifies that len(names) == num_params and that mp_a/mp_b names align
with pos[3] (the true mapping start), regardless of baseline_rg.
"""
import unittest
from unittest.mock import patch


def _make_egh_advanced(n_components, baseline_rg=True, num_baseparams=2):
    """Construct EghAdvansedParams with serial-settings dependencies mocked out."""
    with patch('molass_legacy.ModelParams.EghParams.get_num_baseparams', return_value=num_baseparams), \
         patch('molass_legacy.ModelParams.EghParams.get_setting', return_value=None):
        from molass_legacy.ModelParams.EghParams import EghAdvansedParams
        return EghAdvansedParams(n_components, poresize=None, poreexponent=None, baseline_rg=baseline_rg)


class TestEghAdvansedParamNamesBaselineRgTrue(unittest.TestCase):
    """baseline_rg=True: Rg count = nc + 1 = n_components (the default, backward-compat)."""

    def test_rg_count_in_names_nc3(self):
        p = _make_egh_advanced(3, baseline_rg=True)
        names = list(p.get_parameter_names())
        rg_count = sum(1 for n in names if n.startswith("$R_{g"))
        self.assertEqual(rg_count, p.pos[3] - p.pos[2])

    def test_mp_a_at_pos3_nc3(self):
        p = _make_egh_advanced(3, baseline_rg=True)
        names = list(p.get_parameter_names())
        mp_a_idx = names.index("$mp_a$")
        self.assertEqual(mp_a_idx, p.pos[3],
                         f"mp_a at {mp_a_idx}, expected pos[3]={p.pos[3]}")

    def test_rg_count_in_names_nc2(self):
        p = _make_egh_advanced(2, baseline_rg=True)
        names = list(p.get_parameter_names())
        rg_count = sum(1 for n in names if n.startswith("$R_{g"))
        self.assertEqual(rg_count, p.pos[3] - p.pos[2])

    def test_mp_a_at_pos3_nc2(self):
        p = _make_egh_advanced(2, baseline_rg=True)
        names = list(p.get_parameter_names())
        mp_a_idx = names.index("$mp_a$")
        self.assertEqual(mp_a_idx, p.pos[3])


class TestEghAdvansedParamNamesBaselineRgFalse(unittest.TestCase):
    """baseline_rg=False: Rg count = nc = n_components - 1.

    Previously, get_parameter_names() generated nc+1 Rg names here, shifting
    mp_a/mp_b/UV names by +1 (issue #33).
    """

    def test_rg_count_in_names_nc3(self):
        """Rg name count must equal pos[3] - pos[2] (regression of issue #33)."""
        p = _make_egh_advanced(3, baseline_rg=False)
        names = list(p.get_parameter_names())
        rg_count = sum(1 for n in names if n.startswith("$R_{g"))
        self.assertEqual(rg_count, p.pos[3] - p.pos[2],
                         f"Rg name count {rg_count} != actual Rg param count {p.pos[3]-p.pos[2]} (issue #33)")

    def test_mp_a_at_pos3_nc3(self):
        """Core regression: mp_a name must be at pos[3], not pos[3]+1."""
        p = _make_egh_advanced(3, baseline_rg=False)
        names = list(p.get_parameter_names())
        mp_a_idx = names.index("$mp_a$")
        self.assertEqual(mp_a_idx, p.pos[3],
                         f"mp_a at {mp_a_idx}, expected pos[3]={p.pos[3]} (regression of issue #33)")

    def test_mp_b_at_pos3_plus_1_nc3(self):
        p = _make_egh_advanced(3, baseline_rg=False)
        names = list(p.get_parameter_names())
        mp_b_idx = names.index("$mp_b$")
        self.assertEqual(mp_b_idx, p.pos[3] + 1,
                         f"mp_b at {mp_b_idx}, expected {p.pos[3]+1}")

    def test_rg_count_in_names_nc2(self):
        p = _make_egh_advanced(2, baseline_rg=False)
        names = list(p.get_parameter_names())
        rg_count = sum(1 for n in names if n.startswith("$R_{g"))
        self.assertEqual(rg_count, p.pos[3] - p.pos[2])

    def test_mp_a_at_pos3_nc2(self):
        p = _make_egh_advanced(2, baseline_rg=False)
        names = list(p.get_parameter_names())
        mp_a_idx = names.index("$mp_a$")
        self.assertEqual(mp_a_idx, p.pos[3])


class TestEghAdvansedParamNamesIntegralBaseline(unittest.TestCase):
    """With num_baseparams=3 (integral baseline) names still align."""

    def test_mp_a_at_pos3_baseline_rg_false(self):
        p = _make_egh_advanced(3, baseline_rg=False, num_baseparams=3)
        names = list(p.get_parameter_names())
        mp_a_idx = names.index("$mp_a$")
        self.assertEqual(mp_a_idx, p.pos[3])

    def test_rg_count_matches_pos_baseline_rg_false(self):
        p = _make_egh_advanced(3, baseline_rg=False, num_baseparams=3)
        names = list(p.get_parameter_names())
        rg_count = sum(1 for n in names if n.startswith("$R_{g"))
        self.assertEqual(rg_count, p.pos[3] - p.pos[2])


if __name__ == "__main__":
    unittest.main()
