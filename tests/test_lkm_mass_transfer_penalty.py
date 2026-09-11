"""
Tests for PenaltyUtils.compute_lkm_mass_transfer_penalty (LKM broadening
constraint). Reference values are the real BH/DE fits from SAMPLE5
(molass-papers/experiments/recipe_runner_notebook_redesign.ipynb, Sections
11-12) that motivated this constraint -- component 0 (both runs) and
component 1 (DE run) are the known-pathological, unrealistically broad cases;
the rest are well-behaved.
"""
import pytest
from molass_legacy.Optimizer.PenaltyUtils import compute_lkm_mass_transfer_penalty

# (Pe, t0, R, k_MT) per component, from analysis-007 (BH) and analysis-008 (DE)
ANALYSIS_007 = dict(Pe=12008.879023, t0=434.989961)
ANALYSIS_008 = dict(Pe=12566.938033, t0=432.684173)

PATHOLOGICAL = [
    (ANALYSIS_007, 1.106549, 0.063624),   # 007 comp0
    (ANALYSIS_008, 1.100151, 0.495165),   # 008 comp0
    (ANALYSIS_008, 1.199399, 0.045165),   # 008 comp1
]

WELL_BEHAVED = [
    (ANALYSIS_007, 1.106543, 2.008390),    # 007 comp1
    (ANALYSIS_007, 1.194529, 3.455125),    # 007 comp2
    (ANALYSIS_007, 1.330567, 1670.441937), # 007 comp3
    (ANALYSIS_008, 1.201556, 5.509282),    # 008 comp2
    (ANALYSIS_008, 1.337689, 1037.529537), # 008 comp3
]


@pytest.mark.parametrize("analysis,R,k_MT", PATHOLOGICAL)
def test_pathological_components_are_penalized(analysis, R, k_MT):
    penalty = compute_lkm_mass_transfer_penalty(analysis['Pe'], analysis['t0'], [R], [k_MT])
    assert penalty > 0


@pytest.mark.parametrize("analysis,R,k_MT", WELL_BEHAVED)
def test_well_behaved_components_pass(analysis, R, k_MT):
    penalty = compute_lkm_mass_transfer_penalty(analysis['Pe'], analysis['t0'], [R], [k_MT])
    assert penalty == 0


def test_multi_component_sums_independently():
    Rs = [R for _, R, _ in PATHOLOGICAL[:1] + WELL_BEHAVED[:1]]
    k_MTs = [k for _, _, k in PATHOLOGICAL[:1] + WELL_BEHAVED[:1]]
    combined = compute_lkm_mass_transfer_penalty(
        ANALYSIS_007['Pe'], ANALYSIS_007['t0'], Rs, k_MTs)
    solo = compute_lkm_mass_transfer_penalty(
        ANALYSIS_007['Pe'], ANALYSIS_007['t0'], [Rs[0]], [k_MTs[0]])
    assert combined == pytest.approx(solo)


def test_penalty_decreases_monotonically_with_k_mt():
    analysis = ANALYSIS_007
    R = 1.15
    penalties = [
        compute_lkm_mass_transfer_penalty(analysis['Pe'], analysis['t0'], [R], [k_MT])
        for k_MT in [0.01, 0.1, 1.0, 10.0, 100.0]
    ]
    assert penalties == sorted(penalties, reverse=True)


def test_zero_at_k_mt_infinity_limit():
    penalty = compute_lkm_mass_transfer_penalty(
        ANALYSIS_007['Pe'], ANALYSIS_007['t0'], [1.15], [1e9])
    assert penalty == 0
