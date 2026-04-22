"""
    Solvers.UltraNest.SamplerCallback.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import numpy as np
import sys
from ultranest.viz import get_default_viz_callback
from molass_legacy._MOLASS.Version import is_developing_version

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

    from importlib import reload
    import Solvers.UltraNest.CustomLivePointsWidget
    reload(Solvers.UltraNest.CustomLivePointsWidget)
    from Solvers.UltraNest.CustomLivePointsWidget import CustomLivePointsWidget

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
        if sys.stderr is None:
            sys.stderr = StderrWinmode()    # for sys.stderr.isatty() in ultranest.viz.py on Windows win-app
        if sys.stdout is None:
            sys.stdout = StdoutWinmode()    # for sys.stdout.write(...) in ultranest.integrator.py on Windows win-app

    def __call__(self, points, info, region, transformLayer, region_fresh=False):
        self.default_callback(points, info, region, transformLayer, region_fresh=region_fresh)

        if _DEBUG_CALLBACK:
            print("SamplerCallback.__call__: info.keys()=", info.keys())
            stepsampler_info = info['stepsampler_info']
            if len(stepsampler_info) > 0:
                print("SamplerCallback.__call__: stepsampler_info.keys()=", stepsampler_info.keys())

            if self.sampler.results is not None:
                print("SamplerCallback.__call__: results.keys()=", self.sampler.results.keys())

        self.counter += 1
        if _DEBUG_CALLBACK:
            print("SamplerCallback.__call__: counter=", self.counter)
            print("SamplerCallback.__call__: points.keys()=", points.keys())
            print("SamplerCallback.__call__: points['u'].shape=", points['u'].shape)
            print("SamplerCallback.__call__: points['p'].shape=", points['p'].shape)
            print("SamplerCallback.__call__: points['logl'].shape=", points['logl'].shape)
        m = np.argmax(points['logl'])
        if _DEBUG_CALLBACK:
            print("SamplerCallback.__call__: m=", m)
            print("SamplerCallback.__call__: points['u'][m]=", points['u'][m])
            print("SamplerCallback.__call__: points['p'][m]=", points['p'][m])
        self.callback(points['p'][m], None, False)