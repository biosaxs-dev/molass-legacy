"""
    Solvers.UltraNest.SamplerCallback.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import numpy as np
import sys
from ultranest.viz import get_default_viz_callback
from molass_legacy._MOLASS.Version import is_developing_version

def get_viz_callback():
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

        print("SamplerCallback.__call__: info.keys()=", info.keys())
        stepsampler_info = info['stepsampler_info']
        if len(stepsampler_info) > 0:
            print("SamplerCallback.__call__: stepsampler_info.keys()=", stepsampler_info.keys())
 
        if self.sampler.results is not None:
            print("SamplerCallback.__call__: results.keys()=", self.sampler.results.keys())
        
        self.counter += 1
        print("SamplerCallback.__call__: counter=", self.counter)
        print("SamplerCallback.__call__: points.keys()=", points.keys())
        print("SamplerCallback.__call__: points['u'].shape=", points['u'].shape)
        print("SamplerCallback.__call__: points['p'].shape=", points['p'].shape)
        print("SamplerCallback.__call__: points['logl'].shape=", points['logl'].shape)
        m = np.argmax(points['logl'])
        print("SamplerCallback.__call__: m=", m)
        print("SamplerCallback.__call__: points['u'][m]=", points['u'][m])
        print("SamplerCallback.__call__: points['p'][m]=", points['p'][m])
        self.callback(points['p'][m], None, False)