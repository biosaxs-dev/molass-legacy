"""
    Optimizer.RestartPatcher.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import numpy as np
from matplotlib.widgets import Slider, Button
import molass_legacy.KekLib.DebugPlot as plt
from molass_legacy.SerialAnalyzer.DataUtils import get_in_folder
from .FvScoreConverter import convert_score

class RestartPatcher:
    def __init__(self, caller, optimizer, work_params, jobname):
        self.caller = caller
        self.optimizer = optimizer
        self.work_params = work_params
        self.jobname = jobname

    def preview(self):
        return patch_and_restart_impl(self.caller, self.optimizer, self.work_params, self.jobname)

    def apply(self, restart_params):
        self.caller.start_next_trial(restart_params=restart_params, real_bounds=self.optimizer.real_bounds)

def patch_and_restart_from_editor(editor):
    optimizer = editor.optimizer
    init_params = editor.get_init_params()
    jobname = "---"
    rp = RestartPatcher(editor, optimizer, init_params, jobname)
    ret = rp.preview()
    if ret is not None:
        rp.apply(ret)

def patch_and_restart(caller):
    print("patch_and_restart")
    canvas = caller.canvas
    optimizer = canvas.fullopt
    init_params = canvas.get_current_params()
    job_info = canvas.dialog.get_job_info()
    jobname = job_info[0]
    rp = RestartPatcher(caller, optimizer, init_params, jobname)
    ret = rp.preview()
    if ret is not None:
        rp.apply(ret)

def patch_and_restart_impl(caller, optimizer, init_params, jobname):
    num_components = optimizer.get_num_components()
    in_folder = get_in_folder()
    bounds_array = np.array(optimizer.params_type.get_param_bounds(init_params, real_bounds=optimizer.real_bounds))     # reconsider the need here of real_bounds kwarg
    psinfo = optimizer.get_paramslider_info()
    print("psinfo.cmpparam_names", psinfo.cmpparam_names)
    work_params = init_params.copy()

    def slider_update(k, val):
        print([k], "slider_update", val)
        work_params[k] = val

    def draw_params(params, fig, axes):
        fv = optimizer.objective_func(params)
        sv = convert_score(fv)
        ax1.set_title("UV Decomposition", fontsize=16)
        ax2.set_title("Xray Decomposition", fontsize=16)
        ax3.set_title("Objevive Function Scores in %.1f" % sv, fontsize=16)
        optimizer.objective_func(params, plot=True, axis_info=(fig, axes))

    def add_sliders(component, indeces, fig, slider_axes, sliders, left, width, slider_params, slider_bounds):
        print("add_sliders", component)
        slider_specs = []
        for i, name in enumerate(psinfo.cmpparam_names):
            slider_specs.append((indeces[i], name, *slider_bounds[i], slider_params[i]))

        for k, (j, label, valmin, valmax, valinit) in enumerate(slider_specs, start=0):
            print([k], j, label, component+i)
            ax_ = fig.add_axes([left, 0.4 - 0.05*k, width, 0.03])
            slider  = Slider(ax_, label=label, valmin=valmin, valmax=valmax, valinit=valinit)
            slider.on_changed(lambda val, k_=j: slider_update(k_, val))
            slider_axes.append(ax_)
            sliders.append(slider)

    modelname = optimizer.get_model_name()
    with plt.Dp():
        fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(18,10))
        fig.suptitle("%s Parameter Inspection for Patch and Restart for Job %s in %s" % (modelname, jobname, in_folder), fontsize=20)
        axt = ax2.twinx()
        axt.grid(False)
        axes = (ax1, ax2, ax3, axt)
        draw_params(init_params, fig, axes)

        slider_axes = []
        sliders = []

        def redraw_slider_params(event):
            for ax in axes:
                ax.clear()
            draw_params(work_params, fig, axes)
            fig.canvas.draw_idle()

        def devtest_slider_params(event):
            from importlib import reload
            import Optimizer.Devel.DevMeasureInspect
            reload(Optimizer.Devel.DevMeasureInspect)
            from molass_legacy.Optimizer.Devel.DevMeasureInspect import dev_measure_inspect
            print("devtest_slider_params")
            dev_measure_inspect(optimizer, work_params)

        def try_code_patch(event):
            from importlib import reload
            import Optimizer.Devel.CodePatch
            reload(Optimizer.Devel.CodePatch)
            from molass_legacy.Optimizer.Devel.CodePatch import code_patch
            print("devtest_slider_params")
            current_folder = caller.get_curr_work_folder()      # in case of FullOptDialog
            code_patch(optimizer, work_params, current_folder)

        def reset_slider_params(event):
            nonlocal work_params
            for ax in axes:
                ax.clear()
            work_params = init_params.copy()
            draw_params(work_params, fig, axes)
            for slider in sliders:
                slider.reset()
            fig.canvas.draw_idle()

        def simple_inspect(event):
            from importlib import reload
            import Optimizer.Devel.SimpleInspect
            reload(Optimizer.Devel.SimpleInspect)
            from molass_legacy.Optimizer.Devel.SimpleInspect import simple_inspect_impl
            print("devtest_slider_params")
            simple_inspect_impl(optimizer, work_params)

        for i in range(num_components):
            width = 0.9/num_components
            left = 0.08 + width*i
            slider_width = width*0.6
            indeces = psinfo.get_component_indeces(i)
            print([i], "indeces=", indeces)
            slider_params = work_params[indeces]
            slider_bounds = bounds_array[indeces,:]
            ax = fig.add_axes([left, 0.45, width, 0.03])
            ax.set_axis_off()
            ax.text(0.3, 0.5, "component-%d" % (i+1), ha='center', va='center')
            add_sliders(i, indeces, fig, slider_axes, sliders, left, slider_width, slider_params, slider_bounds)

        for i, (name, j) in enumerate(zip(psinfo.whlparam_names, psinfo.whlparam_indeces)):
            ax_ = fig.add_axes([0.08, 0.2 - 0.05*i, 0.1, 0.03])
            ax_.set_axis_off()
            valmin = bounds_array[j,0]
            valmax = bounds_array[j,1]
            valinit = work_params[j]
            slider  = Slider(ax_, label=name, valmin=valmin, valmax=valmax, valinit=valinit)
            slider.on_changed(lambda val, k_=j: slider_update(k_, val))
            slider_axes.append(ax_)
            sliders.append(slider)

        ax_redraw = fig.add_axes([0.4, 0.05, 0.1, 0.05])
        redraw_button = Button(ax_redraw, 'Redraw')
        redraw_button.on_clicked(redraw_slider_params)
        ax_devtest = fig.add_axes([0.5, 0.05, 0.1, 0.05])
        devtest_button = Button(ax_devtest, 'Devtest')
        devtest_button.on_clicked(devtest_slider_params)
        ax_devtest = fig.add_axes([0.6, 0.05, 0.1, 0.05])
        devtest_button = Button(ax_devtest, 'CodePatch')
        devtest_button.on_clicked(try_code_patch)
        ax_reset = fig.add_axes([0.7, 0.05, 0.1, 0.05])
        reset_button = Button(ax_reset, 'Reset')
        reset_button.on_clicked(reset_slider_params)
        ax_reset = fig.add_axes([0.8, 0.05, 0.1, 0.05])
        reset_button = Button(ax_reset, 'Inspect')
        reset_button.on_clicked(simple_inspect)


        fig.tight_layout()
        fig.subplots_adjust(bottom=0.55)
        ret = plt.show()

    if ret:
        return work_params
    else:
        return None