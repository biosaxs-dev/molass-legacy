<!-- AI Context Standard: v0.1 -->
# AI Assistant Initialization Guide — molass-legacy

**Purpose**: Initialize AI context for navigating this repository  
**Created**: February 19, 2026  
**Magic phrase**: **"Please read COPILOT-INIT.md to initialize"**

> **Note**: This is a legacy codebase. Active development happens in `molass-library`.  
> This repo is a **runtime dependency** of `molass-library` (see `pyproject.toml` there).  
> For the main AI context file, see `molass-library/COPILOT-INIT.md`.

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

This repo is part of the 7-repo VS Code workspace. See `molass-library/COPILOT-INIT.md` Section "Multi-Root Workspace Context" for the full ecosystem map.

---

## 🔄 Updates (AI-Readiness Trail)

| Date | What was learned / added |
|------|--------------------------|
| Feb 19, 2026 | Initial file created. Documented `recognize_peaks` algorithm and known P1+ overlap failure mode. Listed packages most relevant to molass-library cross-repo calls. |

**Principle**: *Never leave this codebase harder to navigate than you found it.*

---

**License**: GNU General Public License v3.0 — Part of molass-legacy
