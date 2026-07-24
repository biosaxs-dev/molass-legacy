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
from collections import namedtuple
import numpy as np

from molass_legacy.Optimizer.FvScoreConverter import convert_score as fv_to_sv
from molass_legacy.Batch.FullBatch import FullBatch


# ---------------------------------------------------------------------------
# Progress-bar mock — absorbs the configure/setitem/getitem calls that the
# estimator 4-stage pipeline makes on editor.pbar without Tkinter.
# ---------------------------------------------------------------------------
class MockProgressBar:
    """Silent ttk.Progressbar replacement for MockEditor."""
    def __init__(self):
        self._state = {'value': 0, 'maximum': 100}
    def configure(self, **kwargs):
        self._state.update(kwargs)
    def __setitem__(self, key, value):
        self._state[key] = value
    def __getitem__(self, key):
        return self._state[key]


# ---------------------------------------------------------------------------
# Upgrade map mirrors PeakEditor._build_library_decomposition._UPGRADE_MAP.
# Kept here so simulate_build_library_decomposition can stay in sync.
# ---------------------------------------------------------------------------
_UPGRADE_MAP = {
    'G1200': ('SDM',  {'pore_dist': 'mono'}),
    'G1300': ('SDM',  {'pore_dist': 'lognormal'}),
    'G1400': ('LKM',  {}),
    'G1500': ('GRM',  {}),
    'G2010': ('CEDM', {}),
    'G2020': ('EDM',  {}),
}

# Fields returned by simulate_build_library_decomposition.
BuildResult = namedtuple('BuildResult', [
    'decomp_egh',       # EGH Decomposition (quick_decomposition result)
    'model_decomp',     # Upgraded Decomposition (or decomp_egh for EGH models; None on failure)
    'ssd_uncorrected',  # Library SSD wrapping uncorrected sd
    'lib_dsets',        # OptDataSets from make_dsets_from_decomposition
    'baseparams',       # [uv_base_array, xr_base_array]
    'basecurves',       # [uv_base_curve, xr_base_curve] — for construct_legacy_optimizer
    'corrected_ssd',    # Library SSD after trimmed_copy().corrected_copy() — for get_lrf_source()
])


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


class MockEditor(FullBatch):
    """
    Minimal editor mock for calling legacy estimators without Tkinter.

    Inherits from FullBatch to gain ``get_lrf_source()`` for the cold estimator
    path (4-stage lognormal pipeline).  When ``corrected_ssd`` is provided,
    FullBatch attributes (``corrected_sd``, ``lrf_src_args1``, ``peak_params_set``,
    ``pre_recog``) are initialised from library objects so that the inherited
    method works without any legacy SerialData.

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
    corrected_ssd : molass.DataObjects.SecSaxsData or None
        Library SSD after ``trimmed_copy().corrected_copy()``.  Used by the
        inherited ``get_lrf_source()`` to provide ``corrected_sd`` (via
        ``SdProxy``) for ``compute_rgs_from_lrf_source``.  Pass
        ``result.corrected_ssd`` from ``build_warm_editor`` / the BuildResult.

    Attributes provided (mirrors PeakEditor interface)
    ---------------------------------------------------
    editor.sd, editor.corrected_sd     -- SdProxy(corrected_ssd) when available, else None
    editor.ecurves                     -- None
    editor.get_n_components()          -- n_components + 1 (includes baseline)
    editor.get_pre_recog_mapping_params()  -- (a, b) UV-XR frame mapping
    editor.baseline_params[1]          -- XR baseline params
    editor.get_uv_base_params()        -- UV baseline params
    editor.decomposition               -- library Decomposition
    editor.dsets                       -- OptDataSets object
    editor.logger                      -- stdlib logger
    editor.peak_params_set             -- PeakParamsSet or [uv_peaks, xr_peaks] list
    """
    def __init__(self, decomposition, dsets, baseparams, model_decomposition=None,
                 ssd_uncorrected=None, corrected_ssd=None, basecurves=None):
        FullBatch.__init__(self)  # sets unified_baseline_type, elution_model, ecurve_info
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

        # Uncorrected SSD for baseparams consistency (molass-legacy#87 pattern).
        # SdmEstimator._estimate_mono and CedmEstimator.estimate_params pass this as
        # data_ssd to make_basecurves_from_decomposition so baseline params are computed
        # from the same (uncorrected) data as the dsets used by the optimizer.
        self._ssd_uncorrected = ssd_uncorrected

        # Progress bar mock — required by estimator 4-stage pipelines
        # (e.g. _estimate_lognormal stages 1-4 call editor.pbar.configure,
        # editor.pbar["value"] = N, editor.update()).
        self.pbar = MockProgressBar()
        self.update = lambda: None

        # Build EGH peak arrays used by peak_params_set and PeakParamsSet.
        if decomposition is not None:
            _xr = np.array([cc.params[:4] for cc in decomposition.xr_ccurves])
            _uv_curves = decomposition.uv_ccurves or []
            _uv = np.array([[cc.get_scale()] for cc in _uv_curves])
            if len(_uv) == 0:
                _uv = np.ones((_xr.shape[0], 1))
        else:
            _xr = np.zeros((3, 4))   # placeholder; caller must set peak_params_set
            _uv = np.ones((3, 1))

        # Initialise FullBatch.get_lrf_source() attributes from library data when
        # corrected_ssd is available.  SdProxy wraps the corrected SSD to provide
        # get_xr_data_separate_ly() — which compute_rgs_from_lrf_source() needs for
        # the Guinier Rg computation — without requiring a legacy SerialData object.
        if corrected_ssd is not None:
            from molass.Bridge.SdProxy import SdProxy
            from molass_legacy.Peaks.PeakParamsSet import PeakParamsSet
            sd_proxy = SdProxy(corrected_ssd)
            self.corrected_sd = sd_proxy   # used by compute_rgs_from_lrf_source
            self.sd = sd_proxy             # also set so LrfSource.__init__ stores it
            (xr_curve, _D), _rg, (uv_curve, _U) = dsets
            baselines = [np.zeros_like(uv_curve.y), np.zeros_like(xr_curve.y)]
            self.lrf_src_args1 = (uv_curve.x, uv_curve.y, xr_curve.x, xr_curve.y, baselines)
            # Pre-populate ecurve_info / baselines so get_curve_xy() returns library
            # data directly without calling get_curve_xy_impl(SdProxy(…)).  This:
            #   - avoids SdProxy compatibility issues with get_curve_xy_impl
            #   - preserves self.baseline_params (from baseparams constructor arg)
            #   - enables get_curve_xy(return_baselines=True) in edit_to_full_sdmparams
            self.ecurve_info = self.lrf_src_args1   # (uv_x, uv_y, xr_x, xr_y, baselines)
            self.baselines = baselines              # same object as lrf_src_args1[4]
            # base_curve_info = (uv_base_curve, uv_baseparams) needed by
            # get_uv_baseline_deprecated → called from edit_to_full_sdmparams
            # (compute_sdm_init_params Stage 1 in cold path).
            if basecurves is not None:
                self.base_curve_info = (basecurves[0], np.array(baseparams[0]))
            else:
                self.base_curve_info = None
            a, b = self.get_pre_recog_mapping_params()
            # PeakParamsSet supports __getitem__ so get_n_components() still works.
            self.peak_params_set = PeakParamsSet(_uv, _xr, a, b)
            self.pre_recog = None
        else:
            # Fallback: plain list compatible with get_n_components() indexing.
            # get_lrf_source() will raise AttributeError if called without corrected_ssd.
            self.base_curve_info = None
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

    def get_uv_base_params(self, xyt=None, debug=False):
        """Returns UV baseline parameter array.
        xyt is accepted but ignored — library baseparams are already fully computed."""
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


# ---------------------------------------------------------------------------
# Warm-start simulation
# ---------------------------------------------------------------------------

def simulate_build_library_decomposition(sd, nc, class_code, verbose=True):
    """Replicate PeakEditor._build_library_decomposition() without Tkinter.

    Produces the *warm-start* state that the real GUI has when any estimator
    is called after the dialog opens — i.e. the state where
    ``editor.model_decomposition`` is already set from a prior ``upgrade()``
    call.  The cold-start (``model_decomposition=None``) is what plain
    ``MockEditor(decomp, dsets, baseparams)`` gives; this function provides
    the complementary warm-start state.

    Typical notebook pattern for catching path-divergence regressions::

        result, editor_warm = build_warm_editor(sd, nc='G1300', class_code=3)
        editor_cold = MockEditor(result.decomp_egh, result.lib_dsets,
                                 result.baseparams)
        # Build optimizers and compare init SV:
        opt_warm = construct_legacy_optimizer(editor_warm, ...)
        opt_cold = construct_legacy_optimizer(editor_cold, ...)
        init_warm = SdmEstimator(editor_warm, ...).estimate_params()
        init_cold = SdmEstimator(editor_cold, ...).estimate_params()
        evaluate_init(opt_warm, init_warm, 'warm (fast path)')
        evaluate_init(opt_cold, init_cold, 'cold (4-stage path)')

    Parameters
    ----------
    sd : molass_legacy SerialData  *or*  molass SecSaxsData (uncorrected)
        Either a legacy ``SerialData`` object or a library ``SecSaxsData``
        wrapping the raw (uncorrected) data (e.g. the result of ``SSD(SAMPLE1)``).
        When a library SSD is passed, ``trimmed_copy().corrected_copy()`` is used
        for the decomposition and the SSD itself becomes ``ssd_uncorrected``.
    nc : int
        Number of elution components.
    class_code : str
        Legacy model class code, e.g. ``'G1300'``, ``'G1200'``, ``'G1400'``.
    verbose : bool, default True
        Print one-line progress messages before each major step so the notebook
        shows the stage being computed rather than appearing to hang silently.

    Returns
    -------
    BuildResult
        Namedtuple with fields ``decomp_egh``, ``model_decomp``,
        ``ssd_uncorrected``, ``lib_dsets``, ``baseparams``.
    """
    import time
    _t0 = time.time()
    def _step(msg):
        if verbose:
            elapsed = time.time() - _t0
            print(f"  [{elapsed:5.1f}s] {msg}", flush=True)

    from molass.Bridge.SdAdapter import make_ssd_from_sd
    from molass.Rigorous.LegacyBridgeUtils import (
        make_dsets_from_decomposition, make_basecurves_from_decomposition)

    if verbose:
        print(f"simulate_build_library_decomposition: {class_code}, nc={nc}", flush=True)

    # Accept legacy SerialData or library SecSaxsData (uncorrected).
    if hasattr(sd, 'trimmed_copy'):
        # Library SecSaxsData passed directly (e.g. SSD(SAMPLE1))
        ssd_uncorrected = sd
        _step("trimmed_copy().corrected_copy()...")
        ssd = sd.trimmed_copy().corrected_copy()
    else:
        # Legacy SerialData
        ssd_uncorrected = make_ssd_from_sd(sd)
        _step("trimmed_copy().corrected_copy()...")
        ssd = ssd_uncorrected.trimmed_copy().corrected_copy()

    _step(f"quick_decomposition(nc={nc})...")
    decomp_egh = ssd.quick_decomposition(num_components=nc)

    _is_egh = class_code in ('G0346', 'G0367')
    if _is_egh:
        model_decomp = decomp_egh
    elif class_code in _UPGRADE_MAP:
        model_name, upgrade_kwargs = _UPGRADE_MAP[class_code]
        # G1300 (SDM lognormal): inject mu_max=ln(3*Rg_max) and sigma=0.05 constraints.
        # Without them the optimizer finds a suboptimal basin (SV≈54 vs ≈68 with constraints).
        # Mirrors the constraint logic in 33f and molass-library#243 / molass-legacy#88.
        if class_code == 'G1300':
            try:
                _rgs = decomp_egh.get_rgs()
                _valid = [float(r) for r in _rgs if r is not None and not np.isnan(float(r)) and float(r) > 0]
                if _valid:
                    _rg_max = max(_valid)
                    _mu_max = float(np.log(3.0 * _rg_max))
                    upgrade_kwargs = dict(upgrade_kwargs)   # don't mutate _UPGRADE_MAP
                    _mp = {'ln_pore_sigma': 0.05, 'mu_max': _mu_max}  # sigma default; overridden below
                    # mu_min and ln_pore_sigma from SerialSettings when available;
                    # fall back to ln(Rg_max) for mu_min, 0.05 for sigma.
                    try:
                        from molass_legacy._MOLASS.SerialSettings import get_setting as _gs
                        _pb = _gs('poresize_bounds')
                        _mp['mu_min'] = float(np.log(_pb[0]))
                        _mp['ln_pore_sigma'] = float(_gs('sdm_pore_sigma'))
                        # Tighten mu_max to poresize_bounds[1]: prevents upgrade NM from
                        # producing poresize > optimizer upper bound, which causes SV=-100
                        # when init params are tested directly via evaluate_init.
                        _mp['mu_max'] = float(np.log(
                            min(np.exp(_mu_max), float(_pb[1]))))
                    except Exception:
                        _mp['mu_min'] = float(np.log(_rg_max))
                    upgrade_kwargs['model_params'] = _mp
                    _step(f"G1300: Rg_max={_rg_max:.1f} Å → mu in [{np.exp(_mp['mu_min']):.1f}, {np.exp(_mu_max):.1f}] Å (molass-library#243)")
            except Exception as _e:
                logging.getLogger(__name__).warning("G1300 mu_max computation failed: %s", _e)
        _step(f"upgrade('{model_name}', ...)...  ← may take a few minutes")
        try:
            model_decomp = decomp_egh.upgrade(model_name, **upgrade_kwargs)
        except Exception as _e:
            logging.getLogger(__name__).warning(
                "simulate_build_library_decomposition: upgrade(%s) failed: %s",
                model_name, _e)
            model_decomp = None
    else:
        model_decomp = None

    _step("get_rg_curve()...  ← Guinier fit on all frames, may take ~30 s")
    rgcurve = ssd.get_rg_curve()
    _step("make_dsets_from_decomposition()...")
    lib_dsets = make_dsets_from_decomposition(
        decomp_egh, rgcurve, data_ssd=ssd_uncorrected)

    _step("make_basecurves_from_decomposition()...")
    decomp_for_base = model_decomp if model_decomp is not None else decomp_egh
    basecurves, baseparams = make_basecurves_from_decomposition(
        decomp_for_base, data_ssd=ssd_uncorrected)

    if verbose:
        print(f"  done ({time.time() - _t0:.1f}s total)", flush=True)
    return BuildResult(decomp_egh, model_decomp, ssd_uncorrected, lib_dsets, baseparams, basecurves, ssd)


def score_model_decomp(result, optimizer, pore_dist='lognormal', t0_upper_bound=None,
                       label='Library upgrade (direct)'):
    """Score the model_decomp params through the fast-path estimator.

    Creates a warm MockEditor (model_decomposition=result.model_decomp), runs the
    SdmEstimator fast path, and calls evaluate_init.  Gives the 'theoretical ceiling':
    the SV achievable if the upgrade quality is good and param extraction is perfect.

    Useful for diagnosing whether a warm-path SV gap is due to (a) upgrade quality
    (constrained vs unconstrained) or (b) estimator extraction logic.

    Parameters
    ----------
    result : BuildResult
        From build_warm_editor or simulate_build_library_decomposition.
    optimizer : BasicOptimizer subclass
        From construct_legacy_optimizer.
    pore_dist : str
        ``'lognormal'`` (G1300) or ``'mono'`` (G1200).
    t0_upper_bound : float or None
        Passed to SdmEstimator.
    label : str
        Display label for the evaluate_init output block.

    Returns
    -------
    sv : float
    xr_params : np.ndarray
    seccol : np.ndarray
    """
    from molass_legacy.Estimators.SdmEstimator import SdmEstimator
    editor = MockEditor(
        result.decomp_egh, result.lib_dsets, result.baseparams,
        model_decomposition=result.model_decomp,
        ssd_uncorrected=result.ssd_uncorrected,
        corrected_ssd=result.corrected_ssd,
        basecurves=result.basecurves,
    )
    editor.fullopt = optimizer
    init_params = SdmEstimator(editor, pore_dist=pore_dist,
                               t0_upper_bound=t0_upper_bound).estimate_params()
    return evaluate_init(optimizer, init_params, label)


def build_warm_editor(sd, nc, class_code, verbose=True):
    """One-stop warm-start MockEditor builder.

    Equivalent to calling ``simulate_build_library_decomposition`` and then
    constructing ``MockEditor`` with the result.  Returns both so the caller
    can inspect individual fields when needed.

    Returns
    -------
    result : BuildResult
    editor : MockEditor
    """
    result = simulate_build_library_decomposition(sd, nc, class_code, verbose=verbose)
    editor = MockEditor(
        result.decomp_egh,
        result.lib_dsets,
        result.baseparams,
        model_decomposition=result.model_decomp,
        ssd_uncorrected=result.ssd_uncorrected,
        corrected_ssd=result.corrected_ssd,
        basecurves=result.basecurves,
    )
    return result, editor
