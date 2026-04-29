"""
Optimizer.DsetsDebug.py

Debugging utilities for comparing parent-process and subprocess OptDataSets
objects.  Addresses molass-legacy#34: the subprocess re-derives its datasets
from disk (via FullOptInput → OptDataSets), which can produce a different
objective landscape from the parent's live datasets — causing the optimizer to
converge to a wrong mp_b even when the parent initialises correctly.

Typical notebook usage
----------------------
::

    from molass_legacy.Optimizer.DsetsDebug import (
        reconstruct_subprocess_dsets,
        compare_dsets,
        plot_dsets_comparison,
        get_mp_b_index,
        sweep_mp_b,
        plot_mp_b_sweep,
    )

    sub_dsets = reconstruct_subprocess_dsets(run_info.work_folder)
    compare_dsets(run_info.dsets, sub_dsets)
    plot_dsets_comparison(run_info.dsets, sub_dsets)

    mp_b_idx = get_mp_b_index(run_info.optimizer)
    mp_b_a, fv_a = sweep_mp_b(run_info.optimizer, run_info.init_params, mp_b_index=mp_b_idx)

    # For subprocess comparison, reconstruct the subprocess optimizer first:
    sub_optimizer = reconstruct_subprocess_optimizer(
        run_info.work_folder,
        n_components=run_info.optimizer.n_components,
        class_code=run_info.optimizer.__class__.__name__,
    )
    sub_optimizer.prepare_for_optimization(run_info.init_params)
    mp_b_b, fv_b = sweep_mp_b(sub_optimizer, run_info.init_params, mp_b_index=mp_b_idx)
    plot_mp_b_sweep([mp_b_a, mp_b_b], [fv_a, fv_b], ['parent', 'subprocess'],
                    init_mp_b=run_info.init_params[mp_b_idx])

Copyright (c) 2026, SAXS Team, KEK-PF
"""
import os
import logging
import numpy as np

logger = logging.getLogger(__name__)

_IN_FOLDER_FILE = "in_folder.txt"


def _get_in_folder_from_work_folder(work_folder):
    """Try to read in_folder from the sentinel file saved in the work folder."""
    path = os.path.join(work_folder, _IN_FOLDER_FILE)
    if os.path.exists(path):
        with open(path, "r") as fh:
            return fh.read().strip()
    return None


def _derive_optimizer_folder(work_folder):
    """Derive optimizer_folder from work_folder.

    Expected layout::

        analysis_folder/
          optimized/           <- optimizer_folder
            treatment.json
            jobs/
              000/             <- work_folder
    """
    return os.path.dirname(os.path.dirname(work_folder))


def reconstruct_subprocess_dsets(work_folder, in_folder=None):
    """Reconstruct the OptDataSets exactly as the subprocess would derive them.

    Replicates the logic of
    ``OptimizerMain.create_optimizer_from_job`` — but runs in the calling
    (parent) process so the result can be compared directly with the
    parent's live datasets.

    Parameters
    ----------
    work_folder : str
        Path to the optimizer job folder (e.g. ``.../optimized/jobs/000``).
        Must contain ``trimming.txt``; ``x_shifts.txt`` is applied if present.
    in_folder : str or None
        Path to the raw data folder.  Falls back to:
        1. ``in_folder.txt`` in *work_folder* (written by BackRunner / InProcessRunner
           when this module's sentinel-file support is active).
        2. ``get_setting('in_folder')`` from the current SerialSettings state.

    Returns
    -------
    dsets : OptDataSets
        Dataset object identical to what the subprocess would receive.
    """
    from molass_legacy._MOLASS.SerialSettings import get_setting, set_setting

    trimming_txt = os.path.join(work_folder, "trimming.txt")
    if not os.path.exists(trimming_txt):
        raise FileNotFoundError(
            f"trimming.txt not found in work_folder={work_folder!r}"
        )

    # Ensure optimizer_folder is set so DataTreatment.load() can find treatment.json.
    optimizer_folder = _derive_optimizer_folder(work_folder)
    set_setting("optimizer_folder", optimizer_folder)
    logger.info("optimizer_folder set to %s", optimizer_folder)

    # Resolve in_folder.
    if in_folder is None:
        in_folder = _get_in_folder_from_work_folder(work_folder)
    if in_folder is None:
        in_folder = get_setting("in_folder")
    if in_folder:
        set_setting("in_folder", in_folder)
        logger.info("in_folder set to %s", in_folder)
    else:
        logger.warning("in_folder could not be determined; relying on current SerialSettings")

    from molass.Bridge.OptimizerInput import OptimizerInput
    fullopt_input = OptimizerInput(in_folder=in_folder, trimming_txt=trimming_txt, legacy=True)
    dsets = fullopt_input.get_dsets()

    x_shifts_file = os.path.join(work_folder, "x_shifts.txt")
    if os.path.exists(x_shifts_file):
        x_shifts = np.loadtxt(x_shifts_file, dtype=int)
        dsets.apply_x_shifts(x_shifts)
        logger.info("applied x_shifts=%s from %s", x_shifts, x_shifts_file)

    return dsets


def reconstruct_subprocess_optimizer(work_folder, n_components, class_code, in_folder=None):
    """Reconstruct the full optimizer as the subprocess would build it.

    This is a heavier version of :func:`reconstruct_subprocess_dsets` that
    also constructs the objective function object — needed for
    :func:`sweep_mp_b`.

    Parameters
    ----------
    work_folder : str
        Optimizer job folder (same as for :func:`reconstruct_subprocess_dsets`).
    n_components : int
        Number of protein components (from ``optimizer.n_components``).
    class_code : str
        Objective function class name (from ``optimizer.__class__.__name__``).
    in_folder : str or None
        Raw data folder; see :func:`reconstruct_subprocess_dsets`.

    Returns
    -------
    optimizer : objective function object (not yet prepared for optimization)
        Call ``optimizer.prepare_for_optimization(init_params)`` before using.
    """
    from molass_legacy.Optimizer.OptimizerMain import create_optimizer_from_job

    trimming_txt = os.path.join(work_folder, "trimming.txt")
    optimizer = create_optimizer_from_job(
        in_folder=in_folder,
        n_components=n_components,
        class_code=class_code,
        trimming_txt=trimming_txt,
    )
    return optimizer


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------

def compare_dsets(dsets_a, dsets_b, label_a="parent", label_b="subprocess"):
    """Print a numerical comparison of two OptDataSets objects.

    Parameters
    ----------
    dsets_a, dsets_b : OptDataSets
    label_a, label_b : str
        Display labels for the two datasets.

    Returns
    -------
    summary : dict
        Key scalar metrics: ``'xr_x_a'``, ``'xr_x_b'``, ``'uv_x_a'``,
        ``'uv_x_b'``, ``'D_shape_a'``, ``'D_shape_b'``, ``'U_shape_a'``,
        ``'U_shape_b'``, and max-difference keys ``'xr_x_maxdiff'``,
        ``'uv_x_maxdiff'``, ``'D_maxdiff'``, ``'U_maxdiff'`` (``None`` when
        shapes differ).
    """
    (xr_a, D_a), rg_a, (uv_a, U_a) = dsets_a.dsets
    (xr_b, D_b), rg_b, (uv_b, U_b) = dsets_b.dsets

    summary = {}

    print(f"=== XR curve ===")
    print(f"  {label_a}: x=[{xr_a.x[0]:.2f}, {xr_a.x[-1]:.2f}], len={len(xr_a.x)}")
    print(f"  {label_b}: x=[{xr_b.x[0]:.2f}, {xr_b.x[-1]:.2f}], len={len(xr_b.x)}")
    summary["xr_x_a"] = (float(xr_a.x[0]), float(xr_a.x[-1]), len(xr_a.x))
    summary["xr_x_b"] = (float(xr_b.x[0]), float(xr_b.x[-1]), len(xr_b.x))

    print(f"\n=== UV curve ===")
    print(f"  {label_a}: x=[{uv_a.x[0]:.2f}, {uv_a.x[-1]:.2f}], len={len(uv_a.x)}")
    print(f"  {label_b}: x=[{uv_b.x[0]:.2f}, {uv_b.x[-1]:.2f}], len={len(uv_b.x)}")
    summary["uv_x_a"] = (float(uv_a.x[0]), float(uv_a.x[-1]), len(uv_a.x))
    summary["uv_x_b"] = (float(uv_b.x[0]), float(uv_b.x[-1]), len(uv_b.x))

    print(f"\n=== XR matrix (D) ===")
    print(f"  {label_a}: shape={D_a.shape}, mean={D_a.mean():.6g}, max={D_a.max():.6g}")
    print(f"  {label_b}: shape={D_b.shape}, mean={D_b.mean():.6g}, max={D_b.max():.6g}")
    summary["D_shape_a"] = D_a.shape
    summary["D_shape_b"] = D_b.shape

    print(f"\n=== UV matrix (U) ===")
    print(f"  {label_a}: shape={U_a.shape}, mean={U_a.mean():.6g}, max={U_a.max():.6g}")
    print(f"  {label_b}: shape={U_b.shape}, mean={U_b.mean():.6g}, max={U_b.max():.6g}")
    summary["U_shape_a"] = U_a.shape
    summary["U_shape_b"] = U_b.shape

    print(f"\n=== Max-absolute differences ===")

    if len(xr_a.x) == len(xr_b.x):
        v = float(np.abs(xr_a.x - xr_b.x).max())
        print(f"  XR x-axis:  {v:.6g}")
        summary["xr_x_maxdiff"] = v
    else:
        print(f"  XR x-axis:  lengths differ ({len(xr_a.x)} vs {len(xr_b.x)})")
        summary["xr_x_maxdiff"] = None

    if len(uv_a.x) == len(uv_b.x):
        v = float(np.abs(uv_a.x - uv_b.x).max())
        print(f"  UV x-axis:  {v:.6g}")
        summary["uv_x_maxdiff"] = v
    else:
        print(f"  UV x-axis:  lengths differ ({len(uv_a.x)} vs {len(uv_b.x)})")
        summary["uv_x_maxdiff"] = None

    if D_a.shape == D_b.shape:
        v = float(np.abs(D_a - D_b).max())
        print(f"  D matrix:   {v:.6g}")
        summary["D_maxdiff"] = v
    else:
        print(f"  D matrix:   shapes differ ({D_a.shape} vs {D_b.shape})")
        summary["D_maxdiff"] = None

    if U_a.shape == U_b.shape:
        v = float(np.abs(U_a - U_b).max())
        print(f"  U matrix:   {v:.6g}")
        summary["U_maxdiff"] = v
    else:
        print(f"  U matrix:   shapes differ ({U_a.shape} vs {U_b.shape})")
        summary["U_maxdiff"] = None

    return summary


def plot_dsets_comparison(dsets_a, dsets_b,
                          label_a="parent", label_b="subprocess",
                          title="DsetsDebug: elution curve comparison"):
    """Four-panel plot comparing XR/UV elution curves from two datasets.

    Top row: raw elution curves overlaid.
    Bottom row: XR vs UV alignment (normalised) for each dataset, so any
    UV-XR offset is immediately visible.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    (xr_a, D_a), rg_a, (uv_a, U_a) = dsets_a.dsets
    (xr_b, D_b), rg_b, (uv_b, U_b) = dsets_b.dsets

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(title)

    # Row 0: raw elution curves
    ax = axes[0, 0]
    ax.plot(xr_a.x, xr_a.y, label=label_a)
    ax.plot(xr_b.x, xr_b.y, "--", alpha=0.7, label=label_b)
    ax.set_title("XR elution curve")
    ax.set_xlabel("frame")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(uv_a.x, uv_a.y, label=label_a)
    ax.plot(uv_b.x, uv_b.y, "--", alpha=0.7, label=label_b)
    ax.set_title("UV elution curve")
    ax.set_xlabel("frame")
    ax.legend(fontsize=8)

    # Row 1: XR vs UV alignment (normalised) — offset between peaks reveals mp_b divergence
    def _norm(y):
        ymax = y.max()
        return y / ymax if ymax > 0 else y

    ax = axes[1, 0]
    ax.plot(xr_a.x, _norm(xr_a.y), label=f"XR ({label_a})")
    ax.plot(uv_a.x, _norm(uv_a.y), "--", alpha=0.8, label=f"UV ({label_a})")
    ax.set_title(f"{label_a}: XR vs UV (normalised)")
    ax.set_xlabel("frame")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.plot(xr_b.x, _norm(xr_b.y), label=f"XR ({label_b})")
    ax.plot(uv_b.x, _norm(uv_b.y), "--", alpha=0.8, label=f"UV ({label_b})")
    ax.set_title(f"{label_b}: XR vs UV (normalised)")
    ax.set_xlabel("frame")
    ax.legend(fontsize=8)

    fig.tight_layout()
    plt.show()
    return fig


# ---------------------------------------------------------------------------
# mp_b sweep
# ---------------------------------------------------------------------------

def get_mp_b_index(optimizer):
    """Return the index of mp_b in the parameter vector.

    Uses ``optimizer.params_type.pos[3]`` (start of the mapping block) + 1,
    because mapping params are ``(mp_a, mp_b)`` in that order.

    Raises ``ValueError`` if the params_type attribute is unavailable.
    """
    try:
        return int(optimizer.params_type.pos[3]) + 1
    except (AttributeError, IndexError, TypeError) as exc:
        raise ValueError(
            "Cannot determine mp_b index from optimizer.params_type.pos[3]; "
            "pass mp_b_index explicitly."
        ) from exc


def sweep_mp_b(optimizer, init_params, mp_b_range=(-200.0, 200.0),
               n_points=81, mp_b_index=None):
    """Evaluate the objective at uniformly spaced mp_b values.

    All other parameters are held at *init_params*.  This produces a 1-D
    cross-section of the objective landscape along the mp_b axis — the key
    diagnostic for #34.

    Parameters
    ----------
    optimizer : objective-function object
        Must expose ``objective_func(params) -> float``.
    init_params : array-like
        Base parameter vector; mp_b is overridden at each evaluation.
    mp_b_range : (float, float)
        Inclusive range of mp_b values to sweep.
    n_points : int
        Number of evaluation points.
    mp_b_index : int or None
        Index of mp_b in the parameter vector.  Auto-detected via
        :func:`get_mp_b_index` if ``None``.

    Returns
    -------
    mp_b_values : ndarray, shape (n_points,)
    fv_values   : ndarray, shape (n_points,)
    """
    params = np.asarray(init_params, dtype=float).copy()
    if mp_b_index is None:
        mp_b_index = get_mp_b_index(optimizer)

    mp_b_values = np.linspace(mp_b_range[0], mp_b_range[1], n_points)
    fv_values = np.empty(n_points)
    for i, mp_b in enumerate(mp_b_values):
        p = params.copy()
        p[mp_b_index] = mp_b
        fv_values[i] = optimizer.objective_func(p)

    return mp_b_values, fv_values


def plot_mp_b_sweep(mp_b_values_list, fv_values_list, labels,
                   init_mp_b=None,
                   title="DsetsDebug: mp_b objective landscape"):
    """Two-panel plot of mp_b sweep results (fv and SV).

    Parameters
    ----------
    mp_b_values_list : list of ndarray
    fv_values_list   : list of ndarray
    labels           : list of str
    init_mp_b        : float or None
        If given, draw a vertical dashed line at this value.
    title : str

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    from molass_legacy.Optimizer.FvScoreConverter import convert_score

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title)

    _colors = ["tab:blue", "tab:red", "tab:green", "tab:orange"]

    for i, (mp_b_values, fv_values, label) in enumerate(
        zip(mp_b_values_list, fv_values_list, labels)
    ):
        c = _colors[i % len(_colors)]
        sv_values = np.array([convert_score(fv) for fv in fv_values])
        best_idx = int(np.argmin(fv_values))

        ax1.plot(
            mp_b_values, fv_values, color=c,
            label=f"{label}  (min fv={fv_values[best_idx]:.4f} @ mp_b={mp_b_values[best_idx]:.1f})"
        )
        ax2.plot(
            mp_b_values, sv_values, color=c,
            label=f"{label}  (max SV={sv_values[best_idx]:.1f} @ mp_b={mp_b_values[best_idx]:.1f})"
        )

    if init_mp_b is not None:
        for ax in (ax1, ax2):
            ax.axvline(init_mp_b, color="gray", linestyle="--",
                       label=f"init mp_b={init_mp_b:.2f}", alpha=0.7)

    ax1.set_xlabel("mp_b")
    ax1.set_ylabel("fv  (lower = better)")
    ax1.set_title("Objective (fv) vs mp_b")
    ax1.legend(fontsize=8)

    ax2.axhline(80, color="green", linestyle=":", alpha=0.5, label="SV=80 (Good)")
    ax2.axhline(60, color="orange", linestyle=":", alpha=0.5, label="SV=60 (Fair)")
    ax2.set_xlabel("mp_b")
    ax2.set_ylabel("SV  (higher = better)")
    ax2.set_title("Score (SV) vs mp_b")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    plt.show()
    return fig
