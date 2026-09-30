# molass_legacy — AI assistant context

This file ships **inside the installed package** (`pip install molass_legacy`),
so it is current for whatever version you have.

Find this file's absolute path at runtime: `python -c "import molass_legacy; print(molass_legacy.context_path())"`
(or `molass_legacy.print_context()` to dump it directly).

If you're developing this repo itself (not just depending on the package),
see `.github/copilot-instructions.md` in the source repository instead —
that covers branching policy, the multi-repo workspace, and dated
investigation history that don't belong here.

---

## What this package is

`molass_legacy` is the original GUI-based MOLASS tool, refactored into a
library that [`molass`](https://pypi.org/project/molass/) (molass-library)
imports at runtime. It contains ~80 sub-packages covering the full SEC-SAXS
analysis pipeline from data loading to 3D reconstruction.

**Read [`molass`'s `CONTEXT.md`](https://github.com/biosaxs-dev/molass-library/blob/main/molass/CONTEXT.md)
first** (also reachable via `python -c "import molass; print(molass.context_path())"`
if it's installed) for the overall architecture, canonical usage, and API
conventions. This file covers only what's specific to `molass_legacy` itself
— most users interact with it indirectly, through `molass.Rigorous`, not
by importing `molass_legacy` directly.

**Key relationship**: `molass.Rigorous` and `molass.LowRank` call into this
package at runtime (`Rigorous/LegacyBridgeUtils.py` on the molass-library
side). `molass_legacy` depends on `molass` too (`pyproject.toml`), so the
two packages must be installed compatibly (see each project's version
constraints) rather than one being strictly "below" the other.

---

## Packages most relevant to callers from `molass`

| Package | What it provides | Called from (`molass`) |
|---------|-----------------|------------------------|
| `QuickAnalysis/ModeledPeaks.py` | `recognize_peaks()`, `get_a_peak()` — peak initialization for decomposition | `molass/LowRank/CurveDecomposer.py` |
| `Models/ElutionCurveModels.py` | `EGH`, `EGHA`, `egh()`, `egha()` — elution curve model functions | Multiple modules |
| `Peaks/ElutionModels.py` | `compute_moments()`, `compute_egh_params()` — moment-based param estimation | `QuickAnalysis/ModeledPeaks.py` |
| `Optimizer/BasicOptimizer.py` | Rigorous optimization engine (`compute_fv`, the objective function base class for all elution models) | `molass/Rigorous/RigorousImplement.py` |
| `Optimizer/NumericalUtils.py` | Small pure helper functions used by the optimizer (e.g. `compute_uv_domain_mask`) — the place to add unit-testable logic extracted from `BasicOptimizer` | `Optimizer/BasicOptimizer.py` |
| `LRF/` | Low-rank factorization internals | `molass/LowRank/` |
| `GuinierAnalyzer/` | Rg estimation (legacy implementation) | `molass/Guinier/` |
| `Mapping/` | Legacy XR/UV mapping GUI + `PeakMapper`/`MappingParams` | `molass/Backward/` (compatibility bridge), `molass/FlowChange/` — primary mapping estimation for `quick_decomposition()`/`trimmed_copy()` is in `molass.Mapping` (molass-library), not here |
| `DataStructure/` | Internal data containers, `LPM.py` | Various |

## Live code reload

Every call to `molass`'s `decomp.score()` / `optimize_rigorously()` reloads
`Optimizer/BasicOptimizer.py` and the relevant `ObjectiveFunctions/*.py`
module (e.g. `G0346.py` for EGH) from disk via `FuncImporter.import_objective_function()`.
Edits here take effect on the next call with no process restart needed — but
this also means those calls aren't cheap to repeat just to "poll" a value.

## Known algorithm limitation: `recognize_peaks` (QuickAnalysis/ModeledPeaks.py)

This is the **default peak initializer** for `molass`'s `quick_decomposition()`.

**Algorithm** (greedy sequential subtraction):
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

**Known failure mode**: at high component overlap, the tallest-peak fit
absorbs signal from both components. After subtraction, the residual is
distorted, so the next peak's initialization is unreliable. Combined with a
single Nelder-Mead run downstream (no multi-start), this produces
high-variance decomposition quality when peaks overlap heavily.

**Workaround**: `molass`'s `quick_decomposition(proportions=[...])` bypasses
`recognize_peaks` entirely, using cumulative-area slicing instead (see
`molass/Decompose/Proportional.py`) — this is the recommended default
whenever components visibly overlap.
