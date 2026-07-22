# coding: utf-8
"""
    GuiSimUtils.py — Tkinter-free GUI simulation utilities

    Provides MockEditor, SimpleLrfSource, and evaluate_init for
    testing and verifying legacy estimators and optimizer initialization
    without opening any Tkinter windows.

    Intended use: Jupyter notebooks that verify GUI consistency during
    the long-term migration of molass-legacy → molass-library.

    See: molass-library/Copilot/refactor/ARCHITECTURE.md

    Copyright (c) 2026, SAXS Team, KEK-PF
"""
import logging
import numpy as np

from molass_legacy.Optimizer.FvScoreConverter import convert_score as fv_to_sv


class SimpleLrfSource:
    """
    Provides pre-computed peak params to get_peak_params_advanced.
    Replaces editor.peak_params_set — no real PeakEditor needed.

    Parameters
    ----------
    decomposition : molass.LowRank.Decomposition
        Library Decomposition object (result of quick_decomposition or upgrade).
    """
    def __init__(self, decomposition):
        # xr_peaks: (nc, 4+) array — get_peak_params_advanced uses first 4 cols
        self.xr_peaks = np.array([cc.params[:4] for cc in decomposition.xr_ccurves])
        # uv_peaks: (nc, 1+) array — get_peak_params_advanced uses peak[0] = height
        if decomposition.uv_ccurves is not None:
            self.uv_peaks = np.array([[cc.get_scale()] for cc in decomposition.uv_ccurves])
        else:
            nc = len(decomposition.xr_ccurves)
            self.uv_peaks = np.ones((nc, 1))


class MockEditor:
    """
    Minimal editor mock for calling legacy estimators without Tkinter.

    Provides exactly the interface that BaseEstimator / EghEstimator /
    SdmEstimator / LkmEstimator expect from a PeakEditor, without
    requiring a SerialData object, a real PeakEditor, or a Tkinter window.

    Parameters
    ----------
    decomposition : molass.LowRank.Decomposition or None
        Library Decomposition object.  Pass None if testing a path where
        the decomposition is expected to be unavailable.
    dsets : OptDataSets
        Prepared dsets from make_dsets_from_decomposition().
    baseparams : list of [uv_base_array, xr_base_array]
        Baseline parameter arrays.  baseparams[0] = UV, baseparams[1] = XR.

    Attributes provided (mirrors PeakEditor interface)
    ---------------------------------------------------
    editor.sd, editor.corrected_sd     -- None (not used by estimation paths)
    editor.ecurves                     -- None
    editor.get_n_components()          -- n_components + 1 (includes baseline)
    editor.get_pre_recog_mapping_params()  -- (a, b) UV-XR frame mapping
    editor.baseline_params[1]          -- XR baseline params
    editor.get_uv_base_params()        -- UV baseline params
    editor.decomposition               -- library Decomposition
    editor.dsets                       -- OptDataSets object
    editor.logger                      -- stdlib logger
    editor.peak_params_set             -- [uv_peaks, xr_peaks] fallback
    """
    def __init__(self, decomposition, dsets, baseparams, model_decomposition=None):
        self.logger = logging.getLogger('MockEditor')
        self.decomposition = decomposition
        self.dsets = dsets
        self.baseline_params = baseparams  # [0]=UV, [1]=XR

        # Required by BaseEstimator.__init__ — not used by any estimation path
        self.sd = None
        self.corrected_sd = None
        self.ecurves = None

        # Model-specific upgraded decomposition (SDM, LKM, EDM, GRM).
        # SdmEstimator._estimate_mono reads editor.model_decomposition to use the
        # library fast path (make_rigorous_initparams).  None → legacy stage-wise path.
        self.model_decomposition = model_decomposition

        # Default peak_params_set (caller may overwrite for specific tests)
        if decomposition is not None:
            _xr = np.array([cc.params[:4] for cc in decomposition.xr_ccurves])
            _uv_curves = decomposition.uv_ccurves or []
            _uv = np.array([[cc.get_scale()] for cc in _uv_curves])
            if len(_uv) == 0:
                _uv = np.ones((_xr.shape[0], 1))
        else:
            _xr = np.zeros((3, 4))   # placeholder; caller must set peak_params_set
            _uv = np.ones((3, 1))
        self.peak_params_set = [_uv, _xr]   # [0]=UV, [1]=XR — matches PeakEditor layout

    def get_n_components(self):
        """Returns number of components + 1 (the baseline 'component')."""
        if self.decomposition is not None:
            return len(self.decomposition.xr_ccurves) + 1
        # Fallback: count from peak_params_set
        return len(self.peak_params_set[1]) + 1

    def get_pre_recog_mapping_params(self):
        """Returns (a, b) UV-XR frame mapping."""
        if self.decomposition is not None:
            return self.decomposition.ssd.get_mapping()
        # Neutral identity-like mapping when decomposition is unavailable
        return 1.0, 0.0

    def get_uv_base_params(self, debug=False):
        """Returns UV baseline parameter array."""
        return np.array(self.baseline_params[0])

    def update_status_bar(self, message):
        """No-op: GuiSimUtils does not show a Tkinter status bar."""
        pass


def evaluate_init(optimizer, init_params, label):
    """
    Evaluate the objective function at init_params and print SV + key params.

    Calls prepare_for_optimization before objective_func so that init_mapping
    and other cached state are set on the optimizer object.

    The score breakdown (return_full=True) is shown for any non-zero term —
    this is the primary diagnostic for catching estimator regressions without
    running a real GUI.  Example: negative_penalty=1087 ⚠️  flags the
    tau/sigma ratio violation that caused SV=-100 in molass-legacy#85.

    Parameters
    ----------
    optimizer : BasicOptimizer or subclass
        Constructed by construct_legacy_optimizer().
    init_params : np.ndarray
        Initial parameter vector.
    label : str
        Display label for the output block.

    Returns
    -------
    sv : float
    xr_params : np.ndarray
    seccol : np.ndarray
    """
    optimizer.prepare_for_optimization(init_params)
    try:
        fv, scores, *_ = optimizer.objective_func(init_params, return_full=True)
    except TypeError:
        fv = optimizer.objective_func(init_params)
        scores = None
    sv = fv_to_sv(fv)
    split = optimizer.split_params_simple(init_params)
    xr_params, xr_base, rgs, mapping, uv_params, uv_base, mr, seccol = split
    print(f"\n=== {label} ===")
    print(f"  fv = {fv:.5f}   SV = {sv:.2f}")
    if scores is not None:
        try:
            names = optimizer.get_score_names()
            nonzero = [(n, v) for n, v in zip(names, scores) if abs(v) > 0.001]
            if nonzero:
                print("  Score breakdown (non-zero terms):")
                for name, val in nonzero:
                    penalty_flag = "  ⚠️" if name in ('negative_penalty', 'order_penalty') and val > 0.1 else ""
                    print(f"    {name}: {val:.4f}{penalty_flag}")
            else:
                print("  Score breakdown: all terms ≈ 0  ✅")
        except AttributeError:
            print(f"  scores: {scores}")
    print(f"  xr_params shape={xr_params.shape}: {xr_params[:8] if xr_params.ndim == 1 else xr_params[:, 2]}")
    print(f"  seccol: {seccol}")
    print(f"  Rg values: {rgs}")
    return sv, xr_params, seccol
