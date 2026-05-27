"""
    Solvers.UltraNest.SamplerCallback.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import logging
import math
import numpy as np
import sys
from ultranest.viz import get_default_viz_callback
from molass_legacy._MOLASS.Version import is_developing_version


def _fv_to_sv(fv):
    return -200.0 / (1.0 + math.exp(-1.5 * fv)) + 100.0

# Set True to re-enable per-outer-iteration diagnostic prints in
# `SamplerCallback.__call__` (was unconditional; silenced to keep the
# notebook output stream clean).
_DEBUG_CALLBACK = False


def _running_in_jupyter_kernel():
    """True when this process is the ipykernel hosting a notebook.

    We use this to skip UltraNest's default viz callback, which creates an
    ipywidgets VBox and updates it via the kernel comm channel on every
    nested-sampling step.  In the subprocess path the kernel doesn't exist
    and the widget is harmless; in the in-process path it adds noticeable
    per-evaluation overhead (each update is a JSON round-trip).
    """
    return 'ipykernel' in sys.modules


def _noop_viz_callback(*args, **kwargs):
    """Drop-in replacement for `get_default_viz_callback()` output.

    UltraNest calls the viz callback with positional args
    `(points, info, region, transformLayer)` and keyword `region_fresh=...`.
    We accept anything and do nothing.
    """
    return None


def get_viz_callback():
    # In-process path inside a Jupyter kernel: skip the widget-based viz
    # entirely (see _running_in_jupyter_kernel for rationale).  The legacy
    # GUI / subprocess path still gets the original behavior because no
    # ipykernel is loaded there.
    if _running_in_jupyter_kernel():
        return _noop_viz_callback

    # In BackRunner subprocess context: skip CustomLivePointsWidget.
    # CustomLivePointsWidget spawns a tkinter GUI subprocess (ProressCanvasClient)
    # whose Queue.put() eventually BLOCKS when the GUI subprocess crashes or
    # its pipe buffer fills up.  SamplerCallback already logs NS progress to
    # optimizer.log, so no viz is needed here.  BackRunner sets this env var.
    # (molass-legacy #67)
    import os as _os
    if _os.environ.get('MOLASS_NS_SUBPROCESS'):
        return _noop_viz_callback

    from importlib import reload
    import molass_legacy.Solvers.UltraNest.CustomLivePointsWidget
    reload(molass_legacy.Solvers.UltraNest.CustomLivePointsWidget)
    from molass_legacy.Solvers.UltraNest.CustomLivePointsWidget import CustomLivePointsWidget

    default_callback = get_default_viz_callback()
    if is_developing_version():
        return CustomLivePointsWidget(default_callback=default_callback)
    else:
        return default_callback

class StderrWinmode:
    def __init__(self):
        pass

    def isatty(self):
        return False
    
    def write(self, *args):
        pass

    def flush(self):
        pass

class StdoutWinmode:
    def __init__(self):
        pass

    def isatty(self):
        return False
    
    def write(self, *args):
        pass

    def flush(self):
        pass

class SamplerCallback:
    def __init__(self, solver, sampler):
        self.default_callback = get_viz_callback()
        self.callback = solver.callback
        self.sampler = sampler
        self.counter = 0
        self.logger = logging.getLogger(__name__)
        if sys.stderr is None:
            sys.stderr = StderrWinmode()    # for sys.stderr.isatty() in ultranest.viz.py on Windows win-app
        if sys.stdout is None:
            sys.stdout = StdoutWinmode()    # for sys.stdout.write(...) in ultranest.integrator.py on Windows win-app

    def __call__(self, points, info, region, transformLayer, region_fresh=False):
        # Guard the entire body: if anything raises (e.g. UltraNest changes the
        # `points` dict structure between Phase 1 and Phase 2, or the terminal
        # viz throws in a subprocess with no tty), UltraNest catches the
        # exception and permanently stops calling viz_callback — silencing all
        # future callback.txt writes.  We therefore protect the minimum
        # guaranteed write first, then attempt the optional parts.
        try:
            self.default_callback(points, info, region, transformLayer, region_fresh=region_fresh)
        except Exception:
            pass  # terminal/widget errors must not prevent the callback.txt write

        self.counter += 1

        # Determine the best live point.  Access 'p' and 'logl' defensively
        # because the Phase-2 SliceSampler may pass a differently-shaped dict.
        best_params = None
        try:
            logl_arr = points['logl']
            m = np.argmax(logl_arr)
            best_params = points['p'][m]

            best_fv = -float(logl_arr[m])
            worst_fv = -float(logl_arr[np.argmin(logl_arr)])
            med_fv = -float(np.median(logl_arr))
            self.logger.info(
                "NS progress: best SV=%.2f, p50 SV=%.2f, worst SV=%.2f (n_live=%d)",
                _fv_to_sv(best_fv), _fv_to_sv(med_fv), _fv_to_sv(worst_fv),
                len(logl_arr),
            )
        except Exception:
            pass  # log failure must not prevent the callback.txt write

        if _DEBUG_CALLBACK:
            try:
                print("SamplerCallback.__call__: counter=", self.counter)
                print("SamplerCallback.__call__: info.keys()=", info.keys())
                print("SamplerCallback.__call__: points.keys()=", points.keys())
            except Exception:
                pass

        # Always write to callback.txt.  Wrap the write and the objective
        # re-evaluation (inside self.callback) in a final try-except so that
        # any error (numerical, I/O, …) cannot propagate out of __call__ and
        # cause UltraNest to permanently disable viz_callback.
        try:
            if best_params is None:
                self.logger.warning(
                    "SamplerCallback: unexpected points structure; "
                    "skipping callback.txt write for counter=%d", self.counter)
            else:
                self.callback(best_params, None, False)
        except Exception:
            pass  # never let __call__ raise