"""
    Optimizer.InProcessRunner.py

    In-process entry point for running an already-prepared optimizer
    inside the parent's Python process — no subprocess, no re-derivation
    of dsets from disk.

    This is the library/notebook path under the split architecture
    (see molass-library/Copilot/DESIGN_split_optimizer_architecture.md).
    The legacy GUI continues to use BackRunner / OptimizerMain via
    subprocess; this module is the in-process counterpart for callers
    that already hold a fully constructed optimizer in memory.

    Copyright (c) 2026, SAXS Team, KEK-PF
"""
import os
import sys
import logging
import numpy as np

from molass_legacy._MOLASS.SerialSettings import get_setting, set_setting
from molass_legacy.KekLib.BasicUtils import mkdirs_with_retry, is_empty_dir
from molass_legacy.Trimming import save_trimming_txt
from .TheUtils import get_optjob_folder_impl
from .FullOptResult import FILES
from .OptimizerUtils import get_impl_method_name, METHOD_NAMES, IMPL_METHOD_NAMES

MAX_NUM_JOBS = 1000


def _ensure_legacy_toplevel_on_syspath():
    """Make `molass-legacy/molass_legacy/` importable as a top-level
    search root.

    Several legacy solver modules use bare top-level imports like
    ``import Solvers.UltraNest.SamplerCallback`` instead of
    ``import molass_legacy.Solvers.UltraNest.SamplerCallback``.  In the
    subprocess path this path is added by side-effect (e.g. when
    ``SerialAnalyzer.DataUtils`` is imported during startup).  In the
    in-process path we may dispatch to a solver before any such
    side-effect has run, so add the directory explicitly and idempotently.
    """
    legacy_pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if legacy_pkg_dir not in sys.path:
        sys.path.insert(0, legacy_pkg_dir)


def _allocate_work_folder():
    """Allocate the next available `<optjob_folder>/NNN/` job folder.

    Mirrors `BackRunner.get_work_folder()` so the in-process path
    produces the same on-disk layout the rest of the tooling
    (`list_rigorous_jobs`, `load_rigorous_result`, callback parsers,
    `MplMonitor` resume) already understands.
    """
    optjob_folder = get_optjob_folder_impl()
    if not os.path.exists(optjob_folder):
        mkdirs_with_retry(optjob_folder)

    for k in range(MAX_NUM_JOBS):
        work_folder = os.path.join(optjob_folder, '%03d' % k)
        if os.path.exists(work_folder):
            if is_empty_dir(work_folder):
                return work_folder
        else:
            mkdirs_with_retry(work_folder)
            return work_folder
    raise RuntimeError("No free job folder available under %s" % optjob_folder)


def _resolve_solver(method, nnn):
    """Map a user-facing method name (or solver code) to the lowercase
    implementation name `optimizer.solve()` expects.

    Accepts:
      * None                 — fall back to `get_impl_method_name(nnn)`
      * 'BH'/'NS'/'MCMC'/'SMC' — user-facing method names
      * 'bh'/'ultranest'/... — already-resolved implementation names
    """
    if method is None:
        return get_impl_method_name(nnn)
    if method in IMPL_METHOD_NAMES:
        return method
    if method in METHOD_NAMES:
        idx = METHOD_NAMES.index(method)
        return get_impl_method_name(nnn, method=idx)
    raise ValueError(
        "Unknown method %r; expected one of %s or %s"
        % (method, METHOD_NAMES, IMPL_METHOD_NAMES)
    )


def run_optimizer_in_process(optimizer, init_params, niter=20, seed=1234,
                             method=None, x_shifts=None, work_folder=None,
                             clear_jobs=True, debug=False):
    """Run an already-prepared optimizer in this process.

    The optimizer is expected to be fully constructed by the caller
    (typically `make_rigorous_decomposition_impl` in molass-library)
    with the library-prepared dsets, base curves, and spectral vectors.
    This function does **not** re-derive any of that — it just performs
    the post-construction work that `OptimizerMain.optimizer_main`
    normally does inside the subprocess:

      1. Allocate / select a job folder.
      2. Persist the same handful of files the subprocess writes
         (init_params.txt, bounds.txt, trimming.txt, x_shifts.txt,
         frozen_components.txt) so existing tooling that reads
         from `<optjob_folder>/NNN/` keeps working.
      3. `chdir` into the job folder so `optimizer.solve()` writes
         `callback.txt` in the expected location.
      4. Dispatch `optimizer.solve(...)` with the resolved solver name.
      5. Restore the original working directory.

    Parameters
    ----------
    optimizer : BasicOptimizer
        Fully constructed optimizer (already wired with dsets / curves).
        Must already have `set_xr_only`, `set_frozen_components`, etc.
        applied as desired by the caller. `prepare_for_optimization`
        does not need to be called in advance — `solve()` calls it.
    init_params : array-like
        Initial parameter vector for the optimization.
    niter : int, optional
        Number of optimizer iterations. Default 20.
    seed : int, optional
        RNG seed for the solver. Default 1234.
    method : str or None, optional
        Solver to dispatch. Accepts user-facing names (`'BH'`, `'NS'`,
        `'MCMC'`, `'SMC'`) or implementation names (`'bh'`, `'ultranest'`,
        `'emcee'`, `'pyabc'`, `'pymc'`). `None` falls back to the
        global `optimization_method` setting.
    x_shifts : array-like or None, optional
        Saved to `x_shifts.txt` for tooling compatibility only. **Not
        applied** to dsets — in the in-process path the dsets are the
        parent's live objects and already carry the correct x-axis.
        (Subprocess re-derives dsets and therefore needs to re-apply.)
    work_folder : str or None, optional
        Pre-allocated job folder. If `None`, a new `<optjob_folder>/NNN/`
        is allocated with the same logic as `BackRunner.get_work_folder()`.
    clear_jobs : bool, optional
        If True (default) and `work_folder` is None, allocate the next
        empty job folder. If False, the caller is responsible for the
        folder choice. Currently informational; allocation always picks
        the next empty slot.
    debug : bool, optional
        If True, additional diagnostic prints from the optimizer.

    Returns
    -------
    result : OptimizeResult
        The return value of `optimizer.solve()` — `result.x` is the
        best parameter vector found, `result.fun` (or `result.nit`
        / `result.nfev` depending on solver) carries solver-specific
        metadata. Caller should consult `callback.txt` in the job
        folder for the per-iteration history.
    work_folder : str
        Absolute path to the job folder where `callback.txt`,
        `optimizer.log`, `init_params.txt`, etc. were written.
        Equivalent to `BackRunner.working_folder` for the subprocess
        path; same shape on disk.
    """
    from molass_legacy.KekLib.ChangeableLogger import Logger

    # 0. Make sure legacy top-level imports (e.g. `import Solvers.*`)
    #    can resolve in this in-process path.
    _ensure_legacy_toplevel_on_syspath()

    # 1. Allocate job folder if not given.
    if work_folder is None:
        work_folder = _allocate_work_folder()
    work_folder = os.path.abspath(work_folder)

    set_setting("optjob_folder", work_folder)
    set_setting("optworking_folder", work_folder)

    nnn = int(os.path.basename(work_folder)[-3:])

    # 2. Persist the files the rest of the tooling expects to find.
    init_params_txt = FILES[2]
    np.savetxt(os.path.join(work_folder, init_params_txt), init_params)

    if optimizer.exports_bounds:
        bounds_txt = FILES[7]
        np.savetxt(os.path.join(work_folder, bounds_txt), optimizer.real_bounds)

    trimming_txt = FILES[6]
    save_trimming_txt(os.path.join(work_folder, trimming_txt))

    if x_shifts is not None:
        x_shifts_txt = FILES[8]
        np.savetxt(os.path.join(work_folder, x_shifts_txt), x_shifts, fmt="%d")

    if getattr(optimizer, 'frozen_components', None) is not None:
        np.savetxt(os.path.join(work_folder, 'frozen_components.txt'),
                   optimizer.frozen_components, fmt="%d")

    with open(os.path.join(work_folder, 'pid.txt'), 'w') as fh:
        fh.write("pid=%d\n" % os.getpid())
    with open(os.path.join(work_folder, 'seed.txt'), 'w') as fh:
        fh.write("seed=%d\n" % seed)

    solver = _resolve_solver(method, nnn)

    parent_logger = logging.getLogger(__name__)
    parent_logger.info(
        "in-process optimizer starting: class=%s, niter=%d, seed=%d, "
        "solver=%s, work_folder=%s",
        optimizer.__class__.__name__, niter, seed, solver, work_folder,
    )

    # 3. chdir into the job folder so solve() writes callback.txt here,
    #    and route a `Logger("optimizer.log")` for parity with the
    #    subprocess `main_impl` log layout.
    saved_cwd = os.getcwd()
    job_logger = None
    try:
        os.chdir(work_folder)
        job_logger = Logger("optimizer.log")
        try:
            from molass_legacy import get_version
            try:
                job_logger.info(get_version(toml_only=True) + " (in-process)")
            except Exception:
                job_logger.info(get_version() + " (in-process)")
        except Exception:
            pass
        job_logger.info("Optimizer started in %s (in-process)", work_folder)
        job_logger.info(
            "class=%s, n_components=%d, solver=%s, niter=%d, seed=%d",
            optimizer.__class__.__name__,
            getattr(optimizer, 'n_components', -1),
            solver, niter, seed,
        )

        # 4. Dispatch.  optimizer.solve() handles prepare_for_optimization,
        #    callback.txt open/close, and solver dispatch.
        #    Disable the GC cycle detector for the duration of the solve: the
        #    optimizer's hot inner loop creates many short-lived arrays and the
        #    cycle collector adds ~25% overhead (measured: 1.33× speedup from
        #    gc.disable in bare fv timing tests).  Reference counting still
        #    runs normally, so memory is freed promptly; only cyclic garbage is
        #    deferred until gc.collect() below.
        import gc as _gc
        _gc.disable()
        try:
            result = optimizer.solve(
                init_params,
                real_bounds=getattr(optimizer, 'real_bounds', None),
                niter=niter,
                seed=seed,
                callback=True,
                method=solver,
                debug=debug,
            )
        finally:
            _gc.enable()
            _gc.collect()

        job_logger.info("in-process optimizer finished")
    finally:
        # 5. Restore cwd unconditionally.
        os.chdir(saved_cwd)

    parent_logger.info("in-process optimizer finished: work_folder=%s", work_folder)
    return result, work_folder
