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
                             clear_jobs=True, debug=False,
                             work_folder_callback=None,
                             stop_event=None):
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
    work_folder_callback : callable or None, optional
        If provided, called with the absolute work_folder path as soon as
        it is allocated — before ``optimizer.solve()`` starts.  Use this
        to notify an async caller (e.g. ``RunInfo``) of the folder path
        without waiting for the full optimization to complete.
        (molass-library issue #132)

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

    # Notify the caller of the work folder as soon as it is known so that
    # async watchers (e.g. MplMonitor watch thread) can start polling
    # callback.txt without waiting for the full optimization to finish.
    # (molass-library issue #132)
    if work_folder_callback is not None:
        try:
            work_folder_callback(work_folder)
        except Exception:
            pass

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

    # Save in_folder so DsetsDebug.reconstruct_subprocess_dsets() can
    # reconstruct the subprocess datasets without live SerialSettings.
    _in_folder = get_setting('in_folder') or ''
    with open(os.path.join(work_folder, 'in_folder.txt'), 'w') as _fh:
        _fh.write(_in_folder)

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
        # Phase 5b: Eagerly remove the StreamHandler(sys.stderr) that Logger.__init__
        # adds to the ROOT logger.  During a kernel restart while the optimizer
        # is still running, logging.shutdown() flushes all root handlers.
        # Flushing the StreamHandler writes to ipykernel's OutStream, which
        # tries to acquire an asyncio lock that is already held during shutdown
        # → deadlock → kernel restart hangs indefinitely. (Phase 5b fix.)
        #
        # Phase 5c: Also register an atexit handler to close the FileHandler
        # (optimizer.log) before logging.shutdown() iterates _handlerList.
        # atexit runs LIFO, so registering here (after logging was imported)
        # guarantees our cleanup fires FIRST, before logging.shutdown().
        # Without this, logging.shutdown() calls fileh.acquire() while the
        # optimizer daemon thread may hold that lock → deadlock and hang.
        try:
            import atexit as _atexit_mod
            import logging as _log_mod
            _root = _log_mod.getLogger()
            _root.removeHandler(job_logger.ch)
            # Phase 5b (fixed): Handler.close() calls _removeHandlerRef(self) but
            # _handlerList contains weakrefs, not handler objects, so the comparison
            # 'handler in [weakref, ...]' is always False — the weakref stays in
            # _handlerList. logging.shutdown() (atexit-registered by Python's own
            # logging module) then dereferences the weakref, gets 'ch', and calls
            # ch.flush() → sys.stderr.flush() → ipykernel OutStream → asyncio lock
            # already held during shutdown → deadlock → kernel restart hangs.
            # Fix: set stream=None FIRST (flush becomes a no-op), then splice
            # _handlerList directly (same pattern as _cleanup_fileh for fileh).
            try:
                job_logger.ch.stream = None   # flush() is now a no-op
            except Exception:
                pass
            try:
                _log_mod._handlerList[:] = [
                    w for w in _log_mod._handlerList if w() is not job_logger.ch
                ]
            except Exception:
                pass
            # Phase 5c: schedule fileh cleanup via atexit.
            # Do NOT call _fh.close() here — it acquires the handler lock,
            # which the daemon optimizer thread may be holding → deadlock.
            # Instead: remove from _handlerList (no lock) + close stream directly.
            def _cleanup_fileh(_jl=job_logger, _r=_root, _lm=_log_mod):
                try:
                    _r.removeHandler(_jl.fileh)
                except Exception:
                    pass
                try:
                    _lm._handlerList[:] = [
                        w for w in _lm._handlerList if w() is not _jl.fileh
                    ]
                except Exception:
                    pass
                try:
                    if hasattr(_jl.fileh, 'stream') and _jl.fileh.stream is not None:
                        _jl.fileh.stream.close()
                        _jl.fileh.stream = None
                except Exception:
                    pass
            _atexit_mod.register(_cleanup_fileh)
            del _cleanup_fileh, _atexit_mod, _root, _log_mod
        except Exception:
            pass
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
        #
        # Run the solver in a daemon thread so the main thread stays
        # interruptible.  VS Code's "Restart Kernel" sends a shutdown message
        # to ipykernel's ZMQ socket (background thread), which then calls
        # PyErr_SetInterrupt() to queue a KeyboardInterrupt in the main thread.
        # That interrupt is only checked between Python bytecodes — but if the
        # main thread is stuck deep inside sampler.run() with back-to-back
        # numpy C calls that hold the GIL, the queued interrupt never gets
        # delivered before VS Code's restart timeout fires, producing a
        # hanging restart and duplicate kernels.
        #
        # By running solve() in a daemon thread and doing a tight join() loop
        # here, the main thread blocks only for 50 ms at a time.  Each
        # join(0.05) releases the GIL, giving Python a safe delivery point for
        # the pending KeyboardInterrupt.  On restart:
        #   - KeyboardInterrupt is raised in the main thread (in join())
        #   - The finally block restores cwd and exits normally
        #   - The daemon thread is killed when the kernel process exits
        import gc as _gc
        import threading as _threading

        _result_holder = [None]
        _exc_holder = [None]

        def _run_solve():
            _gc.disable()
            try:
                _result_holder[0] = optimizer.solve(
                    init_params,
                    real_bounds=getattr(optimizer, 'real_bounds', None),
                    niter=niter,
                    seed=seed,
                    callback=True,
                    method=solver,
                    debug=debug,
                )
            except Exception as _e:
                _exc_holder[0] = _e
            finally:
                _gc.enable()
                _gc.collect()

        # Issue #56: wire stop_event into the optimizer so that minima_callback
        # can return True at the next inter-trial boundary — a cleaner stop than
        # waiting for the ctypes KI to penetrate Nelder-Mead C code.
        if stop_event is not None:
            optimizer._stop_event = stop_event

        _t = _threading.Thread(target=_run_solve, daemon=True, name="aic-active-optimizer")
        _t.start()
        _stop_injected = False
        try:
            while _t.is_alive():
                _t.join(timeout=0.05)   # releases GIL → interrupt delivery point
                if (stop_event is not None and stop_event.is_set()
                        and _t.is_alive() and not _stop_injected):
                    # Inject KeyboardInterrupt into the solver thread so it
                    # exits at the next Python bytecode boundary (~50 ms).
                    # Best-effort: if ctypes is unavailable the flag is set
                    # but the thread runs to completion naturally.
                    try:
                        import ctypes as _ctypes
                        _ctypes.pythonapi.PyThreadState_SetAsyncExc(
                            _ctypes.c_ulong(_t.ident),
                            _ctypes.py_object(KeyboardInterrupt),
                        )
                    except Exception:
                        pass
                    _stop_injected = True
        except KeyboardInterrupt:
            # Kernel restart or user interrupt — let the daemon thread be
            # cleaned up when the process exits; re-raise so callers see it.
            raise

        if _exc_holder[0] is not None:
            raise _exc_holder[0]
        result = _result_holder[0]

        job_logger.info("in-process optimizer finished")
    finally:
        # 5. Explicitly remove Logger handlers from the root logger BEFORE
        #    restoring cwd.  Logger.__init__() adds both a FileHandler
        #    (optimizer.log) and a StreamHandler(sys.stderr) to the ROOT
        #    logger.  If these handlers remain after the daemon thread exits,
        #    Python's logging.shutdown() atexit handler will try to flush them
        #    during kernel restart.  The StreamHandler flush calls
        #    sys.stderr.flush() on ipykernel's OutStream; flushing an OutStream
        #    while the asyncio event loop is shutting down can deadlock the
        #    event loop and cause the kernel restart to hang indefinitely.
        #    (Root cause of the molass-library#139 Phase 5 hang, April 2026.)
        if job_logger is not None:
            try:
                import logging as _logging
                _root = _logging.getLogger()
                _root.removeHandler(job_logger.fileh)
                _root.removeHandler(job_logger.ch)
                # Do NOT call job_logger.fileh.close() here — it acquires
                # the handler lock, but this finally block runs from the
                # optimizer daemon thread which may already hold that lock.
                # Close the underlying stream directly instead.
                try:
                    import logging as _logging_fin
                    _logging_fin._handlerList[:] = [
                        w for w in _logging_fin._handlerList
                        if w() is not job_logger.fileh
                    ]
                except Exception:
                    pass
                try:
                    if hasattr(job_logger.fileh, 'stream') and job_logger.fileh.stream is not None:
                        job_logger.fileh.stream.close()
                        job_logger.fileh.stream = None
                except Exception:
                    pass
            except Exception:
                pass
        # 6. Restore cwd unconditionally.
        os.chdir(saved_cwd)

    parent_logger.info("in-process optimizer finished: work_folder=%s", work_folder)
    return result, work_folder
