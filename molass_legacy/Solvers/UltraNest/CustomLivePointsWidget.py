"""
    Solvers.UltraNest.CustomLivePointsWidget.py

    adapted from ultranest.viz.LivePointsWidget

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import numpy as np
from ultranest.viz import round_parameterlimits, clusteridstrings, spearman

class CustomLivePointsWidget:
    def __init__(self, default_callback=None):
        self.grid = None
        self.label = None
        self.laststatus = None
        self.default_callback = default_callback
    
    def initialize(self, paramnames, width):
        from importlib import reload
        import Solvers.UltraNest.GridspecLayoutMpl
        reload(Solvers.UltraNest.GridspecLayoutMpl)
        from Solvers.UltraNest.GridspecLayoutMpl import GridspecLayoutMpl
        self.grid = GridspecLayoutMpl(len(paramnames), width + 3)
        self.laststatus = []
        for a, paramname in enumerate(paramnames):
            self.laststatus.append('*' * width)

    def __call__(self, points, info, region, transformLayer, region_fresh=False):
        if self.default_callback is not None:
            self.default_callback(points, info, region, transformLayer, region_fresh=region_fresh)

        p = points['p']
        paramnames = info['paramnames']

        plo = p.min(axis=0)
        phi = p.max(axis=0)
        plo_rounded, phi_rounded, paramformats = round_parameterlimits(plo, phi, paramlimitguess=info.get('paramlims'))

        width = 50

        if self.grid is None:
            self.initialize(paramnames, width)
        
        with np.errstate(invalid="ignore"):
            indices = ((p - plo_rounded) * width / (phi_rounded - plo_rounded).reshape((1, -1))).astype(int)
        indices[indices >= width] = width - 1
        indices[indices < 0] = 0
        ndim = len(plo)

        clusterids = transformLayer.clusterids % len(clusteridstrings)
        nmodes = transformLayer.nclusters
        labeltext = ("Mono-modal" if nmodes == 1 else "Have %d modes" % nmodes) + \
            (" | Volume: ~exp(%.2f) " % region.estimate_volume()) + ('*' if region_fresh else ' ') + \
            " | Expected Volume: exp(%.2f)" % info['logvol'] + \
            ('' if 'order_test_correlation' not in info else
             (" | Quality: correlation length: %d (%s)" % (info['order_test_correlation'], '+' if info['order_test_direction'] >= 0 else '-'))
             if np.isfinite(info['order_test_correlation']) else " | Quality: ok")

        if info.get('stepsampler_info', {}).get('num_logs', 0) > 0:
            stepsampler_info = dict(info['stepsampler_info'])
            stepsampler_info['frac_far_enough'] *= 100
            if 'mean_distance' in stepsampler_info:
                labeltext += (
                    "<br/>"
                    'Step sampler performance: %(rejection_rate).1f%% rej/step, %(mean_nsteps)d steps/it'
                    'mean rel jump distance: %(mean_distance).2f (should be >1), %(frac_far_enough).2f%% (should be >50%%)'
                ) % stepsampler_info

        if ndim == 1:
            pass
        elif ndim == 2 and spearman is not None:
            rho, pval = spearman(p)
            if pval < 0.01 and abs(rho) > 0.75:
                labeltext += ("<br/>   %s between %s and %s: rho=%.2f" % (
                    'positive degeneracy' if rho > 0 else 'negative degeneracy',
                    paramnames[0], paramnames[1], rho))
        elif spearman is not None:
            rho, pval = spearman(p)
            for i, param in enumerate(paramnames):
                for j, param2 in enumerate(paramnames[:i]):
                    if pval[i,j] < 0.01 and abs(rho[i,j]) > 0.99:
                        labeltext += ("<br/>   perfect %s between %s and %s" % (
                            'positive relation' if rho[i,j] > 0 else 'negative relation',
                            param2, param))
                    elif pval[i,j] < 0.01 and abs(rho[i,j]) > 0.75:
                        labeltext += ("<br/>   %s between %s and %s: rho=%.2f" % (
                            'positive degeneracy' if rho[i,j] > 0 else 'negative degeneracy',
                            param2, param, rho[i,j]))

        for i, (_param, fmt) in enumerate(zip(paramnames, paramformats)):
            if nmodes == 1:
                line = [' ' for _ in range(width)]
                for j in np.unique(indices[:,i]):
                    line[j] = '*'
                linestr = ''.join(line)
            else:
                line = [' ' for _ in range(width)]
                for clusterid, j in zip(clusterids, indices[:,i]):
                    if clusterid > 0 and line[j] in (' ', '0'):
                        # set it to correct cluster id
                        line[j] = clusteridstrings[clusterid]
                    elif clusterid == 0 and line[j] == ' ':
                        # empty, so set it although we don't know the cluster id
                        line[j] = '0'
                    # else:
                    #     line[j] = '*'
                linestr = ''.join(line)

            oldlinestr = self.laststatus[i]
            for j, (c, d) in enumerate(zip(linestr, oldlinestr)):
                if c != d:
                    if c == ' ':
                        self.grid[i, j + 2] = 0
                    else:
                        self.grid[i, j + 2] = 1

            self.laststatus[i] = linestr
            # self.grid[i,0].value = param
            self.grid[i, 1] = plo_rounded[i]
            self.grid[i,-1] = phi_rounded[i]

        # self.label.value = labeltext
        self.label = labeltext