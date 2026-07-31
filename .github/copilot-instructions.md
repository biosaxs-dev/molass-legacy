<!-- AI Context Standard v0.9.2 - Adopted: 2026-05-07 -->
# AI Assistant Initialization Guide — molass-legacy

**Purpose**: Initialize AI context for navigating this repository  
**Created**: February 19, 2026

> **Note**: This is a legacy codebase. Active development happens in `molass-library`.  
> This repo is a **runtime dependency** of `molass-library` (see `pyproject.toml` there).  
> For the main AI context file, see `molass-library/.github/copilot-instructions.md`.

---

## What This Repo Is

Molass Legacy is the original GUI-based MOLASS tool, refactored into a library that `molass-library` imports at runtime. It contains ~80 sub-packages covering the full SEC-SAXS analysis pipeline from data loading to 3D reconstruction.

**Key relationship**: `molass-library/molass/Rigorous/` and `molass-library/molass/LowRank/` call into this repo. The two repos must be sibling directories for tests to work (see `pythonpath` in `molass-library/pyproject.toml`).

---

## Packages Most Relevant to molass-library

| Package | What it provides | Called from (molass-library) |
|---------|-----------------|------------------------------|
| `QuickAnalysis/ModeledPeaks.py` | `recognize_peaks()`, `get_a_peak()` — peak initialization for decomposition | `molass/LowRank/CurveDecomposer.py` |
| `Models/ElutionCurveModels.py` | `EGH`, `EGHA`, `egh()`, `egha()` — elution curve model functions | Multiple modules |
| `Peaks/ElutionModels.py` | `compute_moments()`, `compute_egh_params()` — moment-based param estimation | `QuickAnalysis/ModeledPeaks.py` |
| `Optimizer/` | Rigorous optimization engine | `molass/Rigorous/RigorousImplement.py` |
| `LRF/` | Low-rank factorization internals | `molass/LowRank/` |
| `GuinierAnalyzer/` | Rg estimation (legacy implementation) | `molass/Guinier/` |
| `DataStructure/` | Internal data containers, `LPM.py` | Various |

---

## Key Algorithm: `recognize_peaks` (QuickAnalysis/ModeledPeaks.py)

This function is the **default peak initializer** for `molass-library`'s decomposition and the root cause of the P1+ overlap failure diagnosed on Feb 19, 2026.

### Algorithm (greedy sequential subtraction)

```
recognize_peaks(x, y, num_peaks, exact_num_peaks):
    y_copy = y.copy()
    for k in range(max_num_peaks):
        params = get_a_peak(x, y_copy)   # fit tallest peak via argmax + Gaussian width scan
        y_model = EGH(x, params)
        y_copy -= y_model                # subtract fitted peak from residual
        peaks_list.append(params)
    return peaks_list
```

### Known failure mode

At high component overlap (≥19% shared area), the tallest-peak fit absorbs signal from both components. After subtraction, the residual is distorted. The second peak's initialization is therefore unreliable. Combined with a single Nelder-Mead run downstream (no multi-start), this produces high variance in decomposition quality.

### Workaround

The `proportions` path in `molass-library` bypasses `recognize_peaks` entirely, using cumulative-area slicing instead. See `molass-library/molass/Decompose/Proportional.py`.

---

## Repository Structure (top-level packages)

```
molass_legacy/
├── QuickAnalysis/    ← recognize_peaks, get_a_peak (critical for molass-library)
├── Models/           ← EGH/EGHA elution models
├── Peaks/            ← Peak detection, ElutionModels
├── Optimizer/        ← Rigorous optimization engine
├── LRF/              ← Low-rank factorization
├── GuinierAnalyzer/  ← Rg estimation
├── DataStructure/    ← Internal data containers
├── DENSS/            ← 3D reconstruction (DENSS wrapper)
├── EFA/              ← Evolving Factor Analysis
├── Baseline/         ← Baseline correction
├── Trimming/         ← Data trimming
├── SecTheory/        ← SEC column theory
├── HdcTheory/        ← HDC theory
├── _MOLASS/          ← Core settings, serial settings
├── molass.py         ← GUI entry point (molass command)
└── [60+ more packages]
```

---

## Testing

```powershell
# Tests require molass-library as sibling directory
pytest tests/ -v
```

---

## Multi-Root Workspace Context

This repo is part of the 7-repo VS Code workspace. See `molass-library/.github/copilot-instructions.md` Section "Multi-Root Workspace Context" for the full ecosystem map.

---

## Response language

**Response language**: English

---

## 🔄 Updates (AI-Readiness Trail)

| Date | What was learned / added |
|------|--------------------------|
| Feb 19, 2026 | Initial file created. Documented `recognize_peaks` algorithm and known P1+ overlap failure mode. Listed packages most relevant to molass-library cross-repo calls. |
| Mar 25, 2026 | Updated to AI Context Standard v0.8; added `init.prompt.md` and `vscode-version.txt` |
| Jun 2026 | **`SdmPlotUtils.plot_objective_state` sec_params length guard** (commit `e315bc5`): The function assumed ≥6 column params (SDM/G1200/G1300 style) and unpacked `t0, rp, N, me, T, mp = sec_params[:6]` unconditionally. LKM (G1400) with ncomp=2 produces 4 params (`num_col_params = 2 + 2*(ncomp-1)`) → `ValueError: not enough values to unpack`. Fix: guard the entire block with `if sec_params is not None and len(sec_params) >= 6:`. Safe because `model_trs` is computed in that block but never used in any plot panel — skipping it for short sec_params has no visual effect. |
| Jun 2026 | **PeakEditor proportional EGH unification** (`PeakEditor.py`, `PeakParamsSet.py`, `EghEstimator.py`): When the legacy GUI dialog opens with the "proportional" option, the initial display and column model estimators should use library EghPeeler (proportional XR decomp) instead of legacy `recognize_peaks`. Three files changed: (1) `PeakParamsSet.__setitem__` added so `peak_params_set[0] = uv_peaks` works; (2) `PeakEditor._build_library_decomposition` now sets `_library_decomp_ready = True` in `finally` block and schedules `_update_display_from_library_decomp` via `self.after(0, ...)` to refresh UV/XR elution panels; (3) `PeakEditor.get_ready_for_optimization` polls `_library_decomp_ready` (every 200ms) before calling `draw_scores()` — ensures the column model estimator always uses library EGH seeds; (4) fallback in `prepare_rg_curve` also sets `_library_decomp_ready = True` so the legacy path doesn't block forever; (5) `EghEstimator.estimate_egh_params` now also injects library UV heights from `decomp.uv_ccurves` (in addition to XR params) so UV initialization is consistent. **Status**: XR panel shows proportional decomp correctly; UV panel still shows non-proportional curves in `draw_scores` — root cause not yet confirmed. Suspected: the `objective_func(init_params, plot=True)` UV display depends on UV weight parameters which may come from a different path (not `init_uv_heights` directly). Resume investigation with added logging in `estimate_egh_params` to confirm `init_uv_heights` values at `draw_scores` time. |
| Jul 2026 | **molass-legacy#85 — draw_scores SV=-100 fix** (`PeakEditor.py`, `G0346.py`, v1.6.12, commit `e0502462`): `FullBatch.construct_optimizer` used `corrected_sd`-based dsets (baseline-subtracted). The legacy data has 6 extra low-q points (q=0.01325..0.01573) not present in library data. After baseline subtraction, these can be negative → Guinier analysis (`compute_fv`) calls `log(negative)` → exception → `fv=BAD_PARAMS_RETURN`. Critically, `prepare_for_optimization` calls `objective_func(..., return_lrf_info=True)` which returns BEFORE Guinier runs → no failure there → GUI logs `valid_components` and `updated rgs` normally. Then `draw_scores` calls `objective_func(..., plot=True)` → Guinier runs → exception → `lrf_info` undefined → `plot_objective_state(lrf_info, ...)` raises NameError → propagates to `draw_scores` → `fv=inf` → SV=-100. **Fix 1** (PeakEditor): Override `construct_optimizer` to use library dsets built from UNCORRECTED `self.sd` (positive at all q) stored as `self._lib_dsets` by `_build_library_decomposition`. **Fix 2** (G0346): Initialize `lrf_info = None` before the try block so NameError no longer occurs when exception is caught. |
| Jul 2026 | **molass-legacy#85 — SV=-100 root cause 2: proportional decomp violates BoundedSecParams bounds** (`PeakEditor.py`, v1.6.12, commit `5888bfc9`): After fixing the Guinier crash (root cause 1), GUI still showed SV=-100. Score breakdown revealed `negative_penalty=1087` and `order_penalty=14.61`. Root cause: `_build_library_decomposition` called `quick_decomposition(proportions=[1]*nc)`. For SAMPLE1 (3 comps), the proportional (cumulative-area-slice) decomp gave `tau[2]=-6.87` with `sigma[2]=8.96` → `\|tau\|/sigma=0.77 > TAU_BOUND_RATIO=0.65` → `negative_penalty=1000*(5.83-6.87)²=1082`. Also `sigma[1]=35.9` (wide middle component) extended past `sigma[2]`'s right boundary → first-come-first-leave violation → `order_penalty=0.01*1461=14.61`. Combined fv≈9.72 → SV=-100. Fix: for EGH (G0346/G0367) only, build the decomp WITHOUT proportions (default `recognize_peaks` path). Default decomp gives `tau[2]≈-0.014`, `sigma[1]≈8.5` → all penalties≈0 → fv≈-1.49 → SV≈81. Confirmed in notebook `33b_gui_simulation.ipynb` cell 19. Column models (SDM/LKM/etc.) still use proportional+upgrade (unchanged). |

| Jul 2026 | **GuiSimUtils + OptimizerUtils bug fixes** (molass-legacy commits `17a8fc81`, `11a30f3d`, `08b62b64`): Created `molass_legacy/Test/GuiSimUtils.py` with `MockEditor`, `SimpleLrfSource`, `evaluate_init` for Tkinter-free GUI simulation. `evaluate_init` uses `return_full=True` so the score breakdown (e.g. `negative_penalty: 1087 ⚠️`) is visible immediately — removes the need for a real GUI run to diagnose init-param regressions. `MockEditor` now accepts `model_decomposition` parameter (read by `SdmEstimator._estimate_mono` fast path). Bug fixes: (1) `OptimizerUtils.get_function_code`: case-insensitive comparison so `'SDM(mono)'` → `'G1200'` (was returning None). (2) `evaluate_init` xr_params display: ndim check for 1D (SDM/LKM/GRM) vs 2D (EGH). Bugs found via `33b_sdm_model.ipynb`; molass-library `LegacyBridgeUtils.construct_legacy_optimizer` also fixed (`model == 'SDM'` → `model.startswith('SDM')`, molass-library commit `318b664`). |

| Jul 2026 | **molass-legacy PeakEditor proportional EGH removed for column models** (`PeakEditor._build_library_decomposition`, commit `464cbee1`): `_build_library_decomposition` was using `proportions=[1]*num_components` for column models (SDM/LKM/GRM/EDM). This caused the second EGH component to widen visibly (covering both neighbor peaks) before the `upgrade()` call. The proportional EGH was then used as the seed for the SDM upgrade → column params N≈379 instead of N≈728 (SV loss ~0.5). Fix: removed the `proportions` kwarg for all models; all models now use non-proportional (default) EGH, which matches the library pipeline (`decomp_egh.upgrade('SDM')`). Confirmed in `molass-researcher/experiments/33_gui_consistency/33b_sdm_model.ipynb` Path C (proportional, 71.60) vs Path D (non-proportional, 72.08). Actual GUI improvement measured: 68.5 → 68.9 (+0.4 SV), consistent with the prediction (+0.48 SV). A residual ~3.2 SV gap remained due to the legacy ssd q-range difference — closed by #86 below. |
| Jul 2026 | **molass-legacy#86 — q-range alignment in `prepare_rg_curve`** (`PeakEditor.py`, commit `fb72745b`): `prepare_rg_curve` was calling `make_ssd_from_corrected_sd(self.corrected_sd)` which wraps corrected SerialData without any trimming → preserves legacy q-range (972 pts, q[0]=0.00631). The library's `trimmed_copy()` clips 6 extra low-q points (q=0.01325..0.01573), giving ~984 pts with q[0]=0.01429. This q-range mismatch shifted the Guinier bisection, degrading SDM init SV by ~3. Fix: replaced with library pipeline `make_ssd_from_sd(self.sd).trimmed_copy().corrected_copy()`; `trimmed=False` / `corrected=False` flags reset so the pipeline runs from scratch. Verified on SAMPLE1: GUI SV improved 68.9 → 71.8 (+2.9 SV). Remaining ~0.3 SV gap closed by #87. |
| Jul 2026 | **molass-legacy#87 — uncorrected baseparams in `_estimate_mono` fast path** (`PeakEditor.py`, `SdmEstimator.py`, commit `7098a0c0`): `_estimate_mono` called `make_basecurves_from_decomposition(model_decomp)` without `data_ssd`, so baseparams were fitted to CORRECTED `model_decomp.ssd` while `_lib_dsets` uses UNCORRECTED data → UV scale differed by ~11%. Fix: (1) `_build_library_decomposition` stores `self._ssd_uncorrected = ssd_uncorrected` after building it for `_lib_dsets`; (2) `_estimate_mono` reads `_ssd_unc = getattr(editor, '_ssd_uncorrected', None)` and passes it as `data_ssd`. `getattr` fallback preserves backward compatibility with `MockEditor`. Verified: GUI SV 71.8 → 72.1 (+0.3). Library reference (Path A) = 72.4; remaining gap ~0.3 is minor q-range difference (legacy 972 vs library 984 pts) — not worth a separate fix. |
| Jul 2026 | **molass-legacy#242 — import cleanup in `_build_library_decomposition`** (`PeakEditor.py`, commit `1d790c29`): Replaced confusingly-named `from molass.Bridge.SdAdapter import make_ssd_from_corrected_sd as _make_ssd_from_sd` with `from molass.Bridge.SdAdapter import make_ssd_from_sd` (neutral alias added in molass-library `db2843a`). The function was called with uncorrected `self.sd` despite the "corrected" name. New name accurately reflects semantics. |
| Jul 2026 | **LkmEstimator `data_ssd` fix** (`LkmEstimator.py`, commit `b11da3a5`): `LkmEstimator.estimate_params()` fast path was missing `data_ssd=_ssd_uncorrected` in `make_basecurves_from_decomposition`. Same root cause as CedmEstimator (commit `2db6bde3`). Fix: `_ssd_unc = getattr(editor, '_ssd_uncorrected', None)`. Verified: SAMPLE1/G1400/LKM Path A = Path B = SV 78.34, ΔSV = 0.000. All model estimator fast paths (SdmEstimator, CedmEstimator, LkmEstimator) now follow the molass-legacy#87 pattern. |
| Jul 2026 | **GrmEstimator `data_ssd` fix** (`GrmEstimator.py`, commit `f2a93605`): `GrmEstimator.estimate_params()` fast path was missing `data_ssd=_ssd_uncorrected` in `make_basecurves_from_decomposition`. Same root cause as all other estimators. Fix: `_ssd_unc = getattr(editor, '_ssd_uncorrected', None)`. Verified: SAMPLE1/G1500/GRM Path A = Path B = SV 78.34, ΔSV = 0.000. All four column-model estimator fast paths (SDM, CEDM, LKM, GRM) now consistently follow the molass-legacy#87 pattern. |
| Jul 2026 | **molass-legacy#89 — EghEstimator degenerate UV height correction** (`EghEstimator.py`, commit `d511e0bd`): For 20230705 (3-comp EGH), library `uv_ccurves[2].get_scale()` returned `-0.0006` (near-zero UV for 3rd component). This caused `consistency_penalty=2.885` → `fv=1.72` → `SV=-86`. Root cause: library UV decomposition occasionally attributes near-zero UV to a component when the UV signal is weak. Fix uses `XrUvScaleRatio` principle: compute `log(uv/xr)` ratios for valid components (uv > `MIN_UV_SCALE=1e-4`); if any component is degenerate, replace its UV height with `xr_scale * exp(mean_log_ratio)`. Logged as "corrected degenerate UV heights at indices [k]". Diagnosis enabled by new `draw_scores` breakdown logging (`ff5efa82`) which immediately showed `consistency_penalty: 2.885` as the dominant term. |
| Jul 2026 | **`SdmEstimator._estimate_lognormal` `data_ssd` fix** (`SdmEstimator.py`, commit `84063fe2`): `_estimate_lognormal` (G1300) fast path was missing `data_ssd=_ssd_uncorrected` in `make_basecurves_from_decomposition` — same root cause as all other estimators. `_estimate_mono` (G1200, line 97) already had this fix since molass-legacy#87. Fix applied: `_ssd_unc = getattr(editor, '_ssd_uncorrected', None)` passed as `data_ssd`. Fast path dispatch: `_estimate_lognormal` checks `column.pore_dist == 'lognormal'` (not `model` — both G1200 and G1300 return `model='sdm'`). Verified: SAMPLE1/G1300 Path A = Path B = SV 33.81, ΔSV = 0.000 ✅ (33f notebook, commit `940594a` in molass-researcher). All five fast paths in SdmEstimator (mono, lognormal), CedmEstimator, LkmEstimator, GrmEstimator now consistently follow the molass-legacy#87 pattern. **Note**: SV=33.81 is poor (expected ~64+ for a good lognormal init) because SAMPLE1's 3-component SDM lognormal is degenerate — components 0 and 1 have near-identical Rg (35 vs 33 Å) → K_SEC values too similar → all 3 components converge to the same column params → overlapping elution curves. This is a model limitation, not a code bug. |
| Jul 2026 | **G1300.py: pre-initialize `lrf_info`/`penalties`/`score_list` before try block** (`G1300.py`, commit `c2a6fff5`): Same bug as G0346 (fixed in molass-legacy#85). `G1300.py` was missing `lrf_info = None` before the try block. When `objective_func(plot=True)` was called with degenerate SDM lognormal params (e.g. SAMPLE1 3-comp case where all components have identical column params), `plot_objective_state` could crash, leaving `lrf_info` undefined → NameError → fv=1e8 → SV=-100 shown in GUI. Fix: added `lrf_info = None` / `penalties = []` / `score_list = [0]*...` before the try block (same pattern as G0346). After fix, GUI will show actual SV (~33) instead of -100. |

| Jul 2026 | **molass-legacy EDM/CEDM model mapping fix** (`CedmEstimator.py`, `GuiSimUtils.py`, commit `2db6bde3`): `CedmEstimator.estimate_params()` fast path called `make_basecurves_from_decomposition(model_decomp)` without `data_ssd` — baseparams computed from corrected SSD while dsets use uncorrected, causing ΔSV=1.37 between library direct (Path A) and CedmEstimator fast path (Path B). Fix: add `_ssd_unc = getattr(editor, '_ssd_uncorrected', None)` and pass `data_ssd=_ssd_unc` (same pattern as #87). Result: ΔSV=0.000 ✅. Also: `MockEditor.__init__` gained `ssd_uncorrected=None` parameter which sets `self._ssd_uncorrected` for use by estimators. Key finding from 33c: G2010=NEDM (EdmParams, 7 params/comp, 41 total), G2020=CEDM (CedmParams, 3 params/comp, 32 total). Library `upgrade('EDM')` returns `model='cedm'` → only G2020 has a library fast path; G2010 always uses legacy path (and requires `poresize` from SerialSettings). |

| Jul 2026 | **EghEstimator init_rgs: use library RgCurve when EghPeeler is active** (`EghEstimator.estimate_egh_params`, commit `ad2d8e60`): When EghPeeler replaces legacy EGH params, `init_xr_params[:,1]` are library frame positions (e.g. 808, 874, 973 for 20230705). The legacy `rg_curve.get_rgs_from_trs()` is a `LegacyRgCurve` built from the narrow legacy ecurve (frames 0–644 for 20230705 where trim start ~605). Frames 808–973 lie beyond that coverage → `add_exclspline` extrapolates → returns ~7 Å. This causes a huge Guinier-deviation penalty since `gdev.rgs` = [51.74, 43.01, 27.95] (from SAXS data) vs `rg_params` = [7, ?, ?] → massive discrepancy → SV=−87.7. Fix: when `decomp` is available (EghPeeler active), use `decomp.ssd.get_rg_curve()` (library RgCurve, full frame range) and interpolate at library peak positions with `np.interp`. Exception fallback to legacy path. Verified: GuiSimUtils init_rgs improved from [59.97, 39.40, 29.62] → [56.42, 41.80, 27.83], t0 from +227 → +170, SV 75.60 → **76.25**. In actual GUI (20230705): expected improvement from SV=−87.7 to ~76. **Key insight**: the bug manifests only when the library peak positions (from EghPeeler, in full-dataset frame space) fall BEYOND the legacy ecurve coverage. For SAMPLE1 the legacy ecurve covers the peak region → no bug. |

| Jul 2026 | **molass-legacy#88 — mono-seeded lognormal init** (`SdmEstimator._estimate_lognormal`, commit `3083cf76`): Stage 3 `estimate_sdm_lognormal_from_monopore` (moment matching) produced SV=33.81 for SAMPLE1 (3 comps, Rg[0]≈Rg[1]) because similar Rg → similar K_SEC → degenerate basin. Fix: replaced moment matching with direct mono-seeding. **T/k semantics asymmetry (key)**: SDM(mono) uses T as Gamma mean; SDM(lognormal) uses T as Gamma scale (mean = k×T) → for matching peak positions, `T_ln = T_mono / k_optimizer` where `k_optimizer=2.0` (optimizer's `k_init` default), NOT `k_mono` (which can be 0.66). **Three constraints**: (1) T_ln = T_mono/2.0, (2) mu_max = ln(3×Rg_max) [via molass-library#243], (3) sigma_init=0.05 (sweet spot, 33g: SV=74.5 vs old default SV=33.81). Also dropped `estimate_sdm_lognormal_from_monopore` from the Stage 1/2/3 imports; Stage 4 call now uses `model_params={'ln_pore_sigma': 0.05, 'mu_max': mu_max_bound}` instead of the kwarg form that was silently ignored. |

| Jul 2026 | **`evaluate_init` length validation** (`GuiSimUtils.py`, commit `a1384f58`, molass-library#244): Added early `ValueError` when `len(init_params) != optimizer.params_type.num_params + NUM_SEC_PARAMS`. Previous behavior: mismatch caused cryptic "not enough values to unpack (expected 6, got 5)" deep in `EghParams.get_extended_bounds`. Root cause of the mismatch: `make_basecurves_from_decomposition` hardcoded `baseline_type=1` while the GUI may have run with `unified_baseline_type=2` (integral baseline, producing 39-param layout instead of 37). Paired fix in `molass-library`: `LegacyBridgeUtils.make_basecurves_from_decomposition` now reads `get_setting('unified_baseline_type')` (commit `ea4d7da`). |

**Principle**: *Never leave this codebase harder to navigate than you found it.*

---

## 🤖 AI-Friendliness Candidates

| Issue | Description | Effort |
|-------|-------------|--------|
| EGH param bounds validation | `_build_library_decomposition` silently produced params violating `BoundedSecParams` (tau/sigma ratio, first-come-first-leave). A pre-flight check logging a warning when any penalty > threshold would surface this immediately instead of after SV=-100. | Small |

---

## 🏗️ Architecture Migration Plans

### SSD-Native Rigorous Optimization Path (planned 2026-07-31)

**Goal**: Eliminate the SD→SSD bridge inside `_build_library_decomposition`. Make PeakEditor's
rigorous optimization path use a pre-built SSD (absolute jv, from the raw data path) instead of
re-deriving SSD from SD on every call.

**Root cause**: `_build_library_decomposition` calls `make_ssd_from_sd(self.sd)`, which inherits
the SD's 0-based jv. Every downstream fix (#86, #87, #89, #244, frame-coord mismatch, UV height
discrepancy) is a symptom of this structural issue.

**Evidence** (33m notebook, 20230705, EGH model):
- Library sim using pre-built SSD (GuiSimUtils): SV = 76.25
- Actual GUI (0-based SD-derived): SV = 70.71
- UV heights: GUI=[0.034, 0.205, 0.565] vs library=[0.015, 0.049, 0.268] (2–4× gap)
- Frame offset: GUI EGH mu=203–363 vs library mu=808–973 (offset ≈ +604 abs frames)

**Implementation (5 steps)**:
1. Find PeakEditor call site in main GUI (FullBatch or similar)
2. At that site, build SSD from raw data: `SSD(folder).trimmed_copy().corrected_copy()`
3. Pass `ssd_corrected` and `ssd_uncorrected` (=trimmed) into `PeakEditor.__init__`
4. In `_build_library_decomposition`: replace `make_ssd_from_sd(self.sd).trimmed_copy().corrected_copy()` → `self._ssd_corrected`
5. Set `self._ssd_uncorrected` from the passed-in value (remove internal re-derivation)

**Scope**: molass-legacy PeakEditor + FullBatch/GUI call site. Naive LRF/Excel stays SD-based.
**Status**: Planned — next implementation target after 33m investigation.

---

**License**: GNU General Public License v3.0 — Part of molass-legacy
