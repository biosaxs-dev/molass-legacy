"""

    Models.ModelUtils.py

    Copyright (c) 2017-2025, SAXS Team, KEK-PF

"""
import re
import numpy as np
from molass_legacy.Decomposer.FitRecord import MIN_SEPARATE_AREA_RATIO, IGNORABLE_AREA_RATIO

model_name_re   = re.compile( r'Model\((\w+)\)' )

def _get_model_name( name ):
    m = model_name_re.match( name )
    if m:
        model_name  = m.group(1).upper()
    else:
        model_name  = None
    return model_name

plot_continue = True

def model_debug_plot(model, params, x_, y_, x, y, tau_hint, where_text, residual, before_params=None, print_callstack=False, work_y=None):
    """
    remember to make friendly this kind of debugging tools
    """
    import molass_legacy.KekLib.DebugPlot as plt
    global plot_continue
    if not plot_continue:
        return

    if print_callstack:
        from molass_legacy.KekLib.CallStack import CallStack
        print('model_debug_plot', CallStack())

    from molass_legacy.KekLib.OurMatplotlib import get_color
    plt.push()

    if before_params is None:
        params_list = [params]
    else:
        params_list = [before_params, params]

    ncols = len(params_list)
    figsize = (8*ncols, 7)
    fig, axes_ = plt.subplots(ncols=ncols, figsize=figsize)

    if ncols == 1:
        axes = [axes_]
        where_texts = [where_text]
    else:
        axes = axes_
        where_texts = ["before fit", "after fit"]

    for ax, params_, where_text_ in zip(axes, params_list, where_texts):
        model_name = model.get_name()
        model_text = "model: %s; " % model_name
        residual_text = " fitting for the residual" if residual else ""
        ax.set_title( model_text + where_text_ + '; tau_hint=' + str(tau_hint) + residual_text )
        ax.plot( x, y, ':', color='orange' )
        ax.plot( x_, y_, color='orange', label='data' )
        if work_y is not None:
            ax.plot(x, work_y, ':', label='work', color='green')
        color=get_color(0)
        ax.plot( x, model.eval( params_, x=x ), ':', color=color )
        ax.plot( x_, model.eval( params_, x=x_ ), color=color, label='model' )

        if model_name in ["EGHA", "EMGA"]:
            mu = params_[1]             # for EMG, EGH
        else:
            assert False
        tau = params_[3]
        ymin, ymax = ax.get_ylim()
        tx = mu
        ty = ymin*0.5 + ymax*0.5
        ax.text(tx, ty, "tau=%.3g" % tau, ha='center', fontsize=30, alpha=0.3)
        ax.legend()

    fig.tight_layout()
    ret = plt.show()
    if not ret:
        plot_continue = False
    plt.pop()

def plot_component_curves(ax, x, y, params_list, baseline, affine=False, color='blue', add_moments=False):
    from molass_legacy.Models.EGH import egh, egha    # placed here to avoid circular import
    model_func = egha if affine else egh
    ax.plot(x, y, color=color, label='data')
    cy_list = []
    for k, params in enumerate(params_list):
        cy = model_func(x, *params)
        cy_list.append(cy)
        ax.plot(x, cy, ':', label='component-%d' % k)
        if add_moments:
            from molass_legacy.Peaks.ElutionModels import compute_moments_from_egh_params
            M = compute_moments_from_egh_params(*params[1:4])
            ax.axvline(x=M[0], color="green")
    ty = np.sum(cy_list, axis=0)
    ax.plot(x, ty, ':', color="red", label='model total')
    ax.plot(x, baseline, color="red")
    ax.legend()

class PairedRangeProxy:
    def __init__(self, range_list):
        self.range_list = range_list
    
    def get_fromto_list(self):
        return self.range_list

def get_paired_ranges_from_params_array(model, x, params_array,
                                        separate_ratio=None,
                                        ignoreable_ratio=IGNORABLE_AREA_RATIO,
                                        return_indeces=False,
                                        want_num_components=None,
                                        select=None,
                                        debug=False):
    if separate_ratio is None:
        separate_ratio = MIN_SEPARATE_AREA_RATIO
    props = model.get_proportions(x, params_array)
    paired_ranges = []
    indeces = []
    if ignoreable_ratio is None or want_num_components is None:
        ignoreable_ratio_ = ignoreable_ratio
    else:
        num_peaks = len(props)
        sorted_props = sorted(props)
        n = num_peaks - want_num_components
        # ignoreable_ratio_ = min(ignoreable_ratio, sorted_props[n]*0.9)
        ignoreable_ratio_ = np.average(sorted_props[n-1:n+1]) if n > 0 else 0.0
        print("sorted_props=", sorted_props)
        print("ignoreable_ratio %g nas been replaced by %g to satisfy want_num_components=%d"
              % (ignoreable_ratio, ignoreable_ratio_, want_num_components) )

    if False:
        with plt.Dp():
            fig, ax = plt.subplots()
            ax.set_title("temp plot")
            ret = plt.show()
            if not ret:
                return

    for k, (params, prop) in enumerate(zip(params_array, props)):
        if select is None:
            if ignoreable_ratio_ is not None and prop < ignoreable_ratio_:
                continue
        else:
            if k not in select:
                continue

        indeces.append(k)
        y = model(x, params)
        m = np.argmax(y)
        max_y = y[m]
        try:
            f, t = [int(round(v)) for v in model.get_range(x, y, max_y=max_y)]
            if prop > separate_ratio:
                top_x = int(round(x[m]))
                range_list = [(f,top_x), (top_x,t)]
            else:
                range_list = [(f,t)]
            paired_ranges.append(PairedRangeProxy(range_list))
        except:
            import logging
            from molass_legacy.KekLib.ExceptionTracebacker import log_exception
            logger = logging.getLogger(__name__)
            log_exception(logger, "get_paired_ranges_from_params_array")
            if debug:
                import molass_legacy.KekLib.DebugPlot as plt
                def get_range_except():
                    from importlib import reload
                    import Models.ModelUtilsException
                    reload(Models.ModelUtilsException)
                    from molass_legacy.Models.ModelUtilsException import get_range_exceptionally
                    return get_range_exceptionally(x, model, params)
                with plt.Dp(extra_button_specs=[("get_range_except", get_range_except)]):
                    fig, ax = plt.subplots()
                    ax.set_title("exception plot")
                    ax.plot(x, y)
                    plt.show()
 
            from molass_legacy.Models.ModelUtilsException import get_range_exceptionally
            f, t = get_range_exceptionally(x, model, params)
            paired_ranges.append(PairedRangeProxy([(f,t)]))
            logger.info("%-th paired range %s has been exceptionally computed", k, (f,t))

    if return_indeces:
        return paired_ranges, indeces
    else:
        return paired_ranges

def compute_cy_list(model, x, params_array):
    cy_list = []
    for params in params_array:
        cy = model(x, params)
        cy_list.append(cy)
    return cy_list

def compute_raw_moment1(x, y):
    return np.sum(x*y)/np.sum(y)    # raw moment

def compute_raw_moments1(x, y_list):
    return np.asarray([np.sum(x*y)/np.sum(y) for y in y_list])

def compute_area_props(cy_list):
    area_props = []
    for cy in cy_list:
        area_props.append(np.sum(cy))
    return np.array(area_props)/np.sum(area_props)

def data_dump(x, y, peaks, prefix=""):
    import os
    from molass_legacy._MOLASS.SerialSettings import get_setting
    temp_folder = get_setting("temp_folder")
    if temp_folder is None:
        temp_folder = os.getcwd()
    print("data_dump: temp_folder=", temp_folder)
    np.savetxt(os.path.join(temp_folder, prefix+"x.dat"), x)
    np.savetxt(os.path.join(temp_folder, prefix+"y.dat"), y)
    np.savetxt(os.path.join(temp_folder, prefix+"peaks.dat"), peaks)