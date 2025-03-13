"""
    Selective/RdrMinimizer.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import numpy as np
from scipy.optimize import minimize, basinhopping
import molass_legacy.KekLib.DebugPlot as plt
from molass_legacy.SerialAnalyzer.DataUtils import get_in_folder
from Selective.V1ParamsAdapter import make_decomp_result_impl

def try_rdr_minimization(advanced_frame, devel=True):
    if devel:
        from importlib import reload
        import Selective.MinimizerUtils
        reload(Selective.MinimizerUtils)
    from .MinimizerUtils import RgComputer

    print("try_rdr_minimization")
    editor = advanced_frame.editor
    editor_frame = editor.get_current_frame()
    model = editor_frame.model
    print("edm_text_from_editor_frame: ", model.get_name(), model.__class__)
    params_array = editor.get_current_params_array()

    fx = editor_frame.fx
    x = editor_frame.x
    y = editor_frame.y
    uv_y = editor_frame.uv_y

    props = model.get_proportions(fx, params_array)
    i = np.argmax(props)    # target component
    print("props=", props, "i=", i)

    paired_ranges = editor_frame.make_restorable_ranges()
    print("paired_ranges=", paired_ranges)
    compute_range = paired_ranges[i]

    D, E, qv, ecurve = editor.sd.get_xr_data_separate_ly()
    num_components = len(params_array)
    rg_computer = RgComputer(model, D, E, qv, fx, y, compute_range, num_components, i)
    C = rg_computer.compute_C(params_array)

    def plot_params(params_array, title, **dp_wkargs):
        rg_list, sg_list = rg_computer.compute_rg_list(C, return_sg=True)
        print("rg_list=", rg_list)
        rdr = abs(rg_list[0] - rg_list[1])*2/(rg_list[0] + rg_list[1])
        with plt.Dp(**dp_wkargs):
            fig, ax = plt.subplots()
            ax.set_title(title, fontsize=20)
            ax.plot(fx, y)
            areas = []
            for cy in C:
                 areas.append(np.sum(cy))
            props = np.array(areas)/np.sum(areas)
            for k, cy in enumerate(C):
                 ax.plot(fx, cy, ":", label="component-%d (%.2g)" % (k, props[k]))
            ty = np.sum(C, axis=0)
            ax.plot(fx, ty, ":", color="red", label="model total")
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            tx = xmin*0.9 + xmax*0.1
            ty = ymin*0.3 + ymax*0.7
            ax.text(tx, ty, "RDR=%.2g" % rdr, fontsize=16)
            ax.legend(fontsize=16)
            fig.tight_layout()
            ret = plt.show()
        return ret

    advanced_frame.progress_update(1)

    plot_params(params_array, "try_rdr_minimization entry")

    rg_list, sg_list = rg_computer.compute_rg_list(C, return_sg=True)
    print("before rg_list=", rg_list)
    sg = sg_list[i]
    gslice = slice(sg.guinier_start, sg.guinier_stop)

    def objective(p, debug=False):
        params_array_ = p.reshape(params_array.shape)
        C_ = rg_computer.compute_C(params_array_)
        ty = np.sum(C_, axis=0)
        try:
            rg_list = rg_computer.compute_rg_list(C_, gslice=gslice)
            rg_ = (rg_list[0] + rg_list[1])/2
        except:
            # numpy.linalg.LinAlgError: SVD did not converge 
            from molass_legacy.KekLib.ExceptionTracebacker import log_exception
            log_exception(None, "try_rdr_minimization.objective: ")
            rg_ = 0
        if rg_ > 10:
            rdr = abs(rg_list[0] - rg_list[1])/rg_
            # scale, N, T, Rg, x0
            # param_rg = params_array_[i,3]
            # fv = np.mean((ty - y)**2) * rdr
            # return fv * max(fv, (rg_ - param_rg)**2)
            ydiv = np.log10(np.mean((ty - y)**2))
            rdiv = np.log10(rdr)
            fv = ydiv*0.95 + rdiv*0.05
            if debug:
                print("rdr=", rdr)
                print("ydiv, rdiv=", ydiv, rdiv, fv)
            return fv
        else:
            fv = np.inf
        return fv

    init_params_array = params_array.flatten()
    objective(init_params_array, debug=True)
    advanced_frame.progress_update(2)

    call_count = 0
    def minima_callback(x, f, accept):
        nonlocal call_count
        call_count += 1
        advanced_frame.progress_update(call_count + 2)

    minimizer_kwargs = dict(method='Nelder-Mead')
    ret = basinhopping(objective, init_params_array, niter=10, callback=minima_callback,
                       minimizer_kwargs=minimizer_kwargs)
    advanced_frame.progress_update(14)

    opt_params_array = ret.x.reshape(params_array.shape)
    objective(opt_params_array, debug=True)
    optC = rg_computer.compute_C(opt_params_array)
    rg_list = rg_computer.compute_rg_list(optC, gslice=gslice)
    print("after rg_list=", rg_list)
    advanced_frame.progress_update(15)
    dp_kwargs = dict(window_title="RDR minimization", 
                     button_spec=["Accept", "Cancel"], 
                     guide_message="If you accept, the model will be updated with the new parameters.")
    ret = plot_params(opt_params_array, "Minimized RDR on %s" % get_in_folder(), **dp_kwargs)
    if ret:
        advanced_frame.progress_update(16)
        decomp_result = make_decomp_result_impl(editor, opt_params_array)
        advanced_frame.progress_update(17)
        advanced_frame.update_button_status(change_id="RDR")
        editor.update_current_frame_with_result(decomp_result)
        advanced_frame.progress_update(18)
    else:
        advanced_frame.progress_update(18)
    advanced_frame.progress_update(-1)