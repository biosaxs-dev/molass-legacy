"""
    DecompPlotUtils.py

    Copyright (c) 2023-2024, SAXS Team, KEK-PF
"""
import copy
import numpy as np
from molass_legacy.KekLib.OurMatplotlib import get_color
from DevSettings import get_dev_setting

def draw1_impl(self):
    ax  = self.ax1
    ax.cla()
    fx  = self.fx
    x   = self.x
    if self.elution_fig_type == 0:
        y   = self.y
        color = 'orange'
        opt_recs = self.opt_recs
    else:
        y   = self.uv_y
        color = 'blue'
        opt_recs = self.opt_recs_uv

    self.the_curves = []

    label = 'data'
    ax.plot(x, y, color=color, label=label)
    self.the_curves.append([label, x, y])

    total_y = np.zeros(len(y))
    resid_y = copy.deepcopy(y)

    cy_list = []
    areas = []
    for k, rec in enumerate(opt_recs):
        func = rec[1]
        y_ = func(fx)
        cy_list.append(y_)
        areas.append(np.sum(y_))
        self.the_curves.append([label, x, y_])
        total_y += y_
        resid_y -= y_

    props = np.array(areas)/np.sum(areas)
    for k, y_ in enumerate(cy_list):
        label = 'component-%d, (%.2f)' % ((k+1), props[k])
        ax.plot(x, y_, ':', label=label, color=get_color(k), linewidth=3)

    label = 'model total'
    ax.plot(x, total_y, ':', label=label, color='red', linewidth=3)
    self.the_curves.append([label, x, total_y])

    ax.legend()
    self.fig1_range_parts = []
    self.add_range_patchs(ax, self.fig1_range_parts)

    # draw the residual
    ax1r = self.ax1r
    ax1r.cla()
    label = 'residual'
    ax1r.plot(x, resid_y, label=label, color='gray')
    self.the_curves.append([label, x, resid_y])

    ymin, ymax = ax.get_ylim()
    yminr, ymaxr = ax1r.get_ylim()
    ymidr = (yminr + ymaxr)/2
    half_height_r = max((ymax - ymin)*self.height_ratio, ymaxr - yminr)/2
    ax1r.set_ylim(ymidr - half_height_r, ymidr + half_height_r)

    ax1r.legend(bbox_to_anchor=(1, 1), loc='lower right')
    self.fig1_range_parts_resid = []
    self.add_range_patchs(ax1r, self.fig1_range_parts_resid)

    self.fit_error = np.sum(np.abs(resid_y))

    if get_dev_setting('running_with_tester'):
        self.write_residual_amount_to_testerlog(y, resid_y)
