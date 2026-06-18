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

**Principle**: *Never leave this codebase harder to navigate than you found it.*

---

**License**: GNU General Public License v3.0 — Part of molass-legacy
