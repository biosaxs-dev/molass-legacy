"""
    Elution.Proportional.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import numpy as np
from scipy.optimize import minimize
import molass_legacy.KekLib.DebugPlot as plt
from molass_legacy.KekLib.ExceptionTracebacker import log_exception

def decompose_by_binary_props(x, y, model, p1, p2, debug=False):
    # plot_info = xy_for_plot if debug else None
    plot_info = None
    params_array = model.guess_binary_peaks(x, y, p1, p2, debug=debug, plot_info=plot_info)
    target_props = np.array([p1, p2])

    def objective(p):
        areas = []
        cy_list = []
        for params in p.reshape(params_array.shape):
            cy = model(x, params)
            cy_list.append(cy)
            areas.append(np.sum(cy))
        try:
            ty = np.sum(cy_list, axis=0)
            props = np.array(areas)/np.sum(areas)
            fv = max(np.sum((ty - y)**2), np.sum((props - target_props)**2))
        except:
            # 
            fv = np.inf
        return fv

    ret = minimize(objective, params_array.flatten())
    params_array = ret.x.reshape(params_array.shape)
    return params_array

def fetch_component_by_prop(x, y, model, p, debug=False):
    params_array = model.guess_binary_peaks(x, y, p, 1-p, debug=False)
    return params_array[0]

def decompose_by_proportions_impl(x, y, model, target_props, debug=False):
    global xy_for_plot
    if debug:
        xy_for_plot = (x, y)
    sum_props = np.sum(target_props)
    if abs(sum_props - 1.0) > 1e-6:
        print("decompose_by_proportions: sum of target_props is not 1.0, they will be normalized.")
        target_props = target_props/sum_props

    resid_y = y.copy()
    params_list = []
    num_components = len(target_props)
    for j in range(num_components-1):
        p1 = target_props[j]
        p2 = np.sum(target_props[j+1:])
        total_p = p1 + p2
        p1 = p1/total_p
        p2 = p2/total_p
        print([j], p1, p2)
        try:
            if j < num_components-2:
                params = fetch_component_by_prop(x, resid_y, model, p1, debug=debug)
            else:
                params_array = decompose_by_binary_props(x, resid_y, model, p1, p2, debug=debug)
                params = params_array[0]
            cy = model(x, params)
            resid_y -= cy
            params_list.append(params)
        except:
            log_exception(None, "decompose_by_proportions: failed in %dth decopose_by_binary_props." % j)
        
    params_list.append(params_array[1])
    temp_params_array = np.array(params_list)

    topxv = temp_params_array[:,1]
    init_params = temp_params_array[:,[0,2,3,4]]

    def final_objective(p, title=None):
        areas = []
        cy_list = []
        for k, params in enumerate(p.reshape(init_params.shape)):
            params_ = np.array([params[0], topxv[k], *params[1:]])
            cy = model(x, params_)
            cy_list.append(cy)
            areas.append(np.sum(cy))
        try:
            ty = np.sum(cy_list, axis=0)
            props = np.array(areas)/np.sum(areas)
            fv = max(np.sum((ty - y)**2), np.sum((props - target_props)**2))
            if title is not None:
                print(title, fv)
                with plt.Dp():
                    fig, ax = plt.subplots()
                    ax.set_title(title)
                    ax.plot(x, y)
                    for cy in cy_list:
                        ax.plot(x, cy, ":")
                    ax.plot(x, ty, ":", color="red")
                    fig.tight_layout()
                    plt.show()
        except:
            #
            fv = np.inf
        return fv

    init_params_ = init_params.flatten()
    if debug:
        final_objective(init_params_, title="before final_objective")
    res = minimize(final_objective, init_params_, method="Nelder-Mead")
    if debug:
        final_objective(res.x, title="after final_objective")
    opt_params_array = res.x.reshape(init_params.shape)
    ret_paramd_list = []
    for k, params in enumerate(opt_params_array):
        params_ = np.array([params[0], topxv[k], *params[1:]])
        ret_paramd_list.append(params_)
    ret_params_array = np.array(ret_paramd_list)

    if debug:
        with plt.Dp():
            fig, ax = plt.subplots()
            ax.set_title("decompose_by_proportions_impl")
            ax.plot(x, y)
            for params in ret_params_array:
                cy = model(x, params)
                px, py = model.get_peaktop_xy(x, params)
                ax.plot(x, cy, ":")
                ax.plot(px, py, "o", color="red")
            fig.tight_layout()
            plt.show()
    
    return ret_params_array

def decompose_by_proportions(fx, y, uv_y, model, target_props, xr_only=True, ask_return=False, debug=False):
    if debug:
        with plt.Dp():
            fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,5))
            fig.suptitle("decompose_by_proportions: input", fontsize=20)
            ax1.plot(fx, uv_y)
            ax2.plot(fx, y)
            fig.tight_layout()
            plt.show()

    try:
        params_array_xr = decompose_by_proportions_impl(fx, y, model, target_props, debug=debug)
    except:
        log_exception(None, "decompose_by_proportions: failed to decompose XR", n=10)
        params_array_xr = None

    if xr_only:
        return params_array_xr
    
    try:
        params_array_uv = decompose_by_proportions_impl(fx, uv_y, model, target_props, debug=False)
    except:
        log_exception(None, "decompose_by_proportions: failed to decompose UV", n=10)
        params_array_uv = None

    assert not (params_array_xr is None and params_array_uv is None)
    if params_array_xr is None:
        params_array_xr = model.guess_from_the_other(fx, y, uv_y, params_array_uv)
    else:
        params_array_uv = model.guess_from_the_other(fx, uv_y, y, params_array_xr)

    if ask_return:
        dp_kwargs = dict(window_title="KPD minimization", 
                        button_spec=["Accept", "Cancel"], 
                        guide_message="If you accept, the model will be updated with the new parameters.")
        with plt.Dp(**dp_kwargs):
            fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,5))
            fig.suptitle("Change Num Components", fontsize=20)
            ax1.plot(fx, uv_y)
            cy_list = []
            for params in params_array_uv:
                cy = model(fx, params)
                cy_list.append(cy)
                ax1.plot(fx, cy, ":")
            ty = np.sum(cy_list, axis=0)
            ax1.plot(fx, ty, ":", color="red")

            ax2.plot(fx, y)
            cy_list = []
            for params in params_array_xr:
                cy = model(fx, params)
                cy_list.append(cy)
                ax2.plot(fx, cy, ":")
            ty = np.sum(cy_list, axis=0)
            ax2.plot(fx, ty, ":", color="red")
            fig.tight_layout()
            ret = plt.show()
    if not ask_return or ret:
        return params_array_xr, params_array_uv
    else:
        return None