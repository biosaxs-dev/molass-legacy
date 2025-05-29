"""
    Selective.LrfSource.py

    Copyright (c) 2024-2025, SAXS Team, KEK-PF
"""
import os
import logging
import numpy as np
from importlib import reload
import molass_legacy.KekLib.DebugPlot as plt
from molass_legacy._MOLASS.SerialSettings import get_setting, setting_file, save_settings
from molass_legacy.Models.ElutionCurveModels import EGH, EGHA
from molass_legacy.Models.ModelUtils import compute_area_props

NUM_MONOPOREPARAMS = 6

class LrfSource:
    def __init__(self, sd, corrected_sd, lrf_src_args1, peak_params_set, pre_recog=None):
        self.logger = logging.getLogger(__name__)
        self.sd = sd
        self.corrected_sd = corrected_sd
        uv_x, uv_y, xr_x, xr_y, baselines = lrf_src_args1
        self.uv_x = uv_x
        self.uv_y = uv_y
        self.xr_x = xr_x
        self.xr_y = xr_y
        self.baselines = baselines
        self.peak_params_set = peak_params_set
        self.uv_peaks = peak_params_set.uv_peaks
        self.xr_peaks = peak_params_set.xr_peaks
        if self.xr_peaks.shape[1] == 4:
            self.model = EGH()
        else:
            self.model = EGHA() 
        self.rg_info = None             # _compute_rgs call will be delayed
        self.egh_moments_list = None
        self.pre_recog = pre_recog
   
    def _compute_rgs(self,
                     keep_num_components=False,
                     want_num_components=False,
                     select=None, debug=False):
        if debug:
            import molass_legacy.Selective.LrfRgComputer
            reload(molass_legacy.Selective.LrfRgComputer)
        from .LrfRgComputer import compute_rgs_from_lrf_source
        if debug:
            print("Computing Rgs...")
        rgs, trs, orig_props, peak_rgs, peak_trs, props, indeces, qualities = compute_rgs_from_lrf_source(self,
                                                                                                          keep_num_components=keep_num_components,
                                                                                                          want_num_components=want_num_components,
                                                                                                          select=select)
        self.num_components_kept = keep_num_components
        if debug:
            def cb(ax):
                axt = ax.twinx()
                axt.grid(False)
                axt.plot(trs, rgs, 'o', color='green')
            self.draw(cb=cb, extra=False)
        self.rg_info = rgs, trs, orig_props, peak_rgs, peak_trs, props, indeces, qualities

    def get_peaks(self):
        if self.rg_info is None:
            # note that this is intended to be used without any Rg info filtering
            return self.xr_peaks
        else:
            indeces = self.rg_info[6]
            return self.xr_peaks[indeces]

    def get_egh_moments_list(self):
        from molass_legacy.Models.Stochastic.MomentUtils import compute_egh_moments
        if self.egh_moments_list is None:
            peaks = self.get_peaks()
            self.egh_moments_list = compute_egh_moments(peaks)
        return self.egh_moments_list

    def compute_rgs(self, keep_num_components=False, want_num_components=None, select=None, debug=False):
        if self.rg_info is None:
            self._compute_rgs(keep_num_components=keep_num_components,
                              want_num_components=want_num_components,
                              select=select)
        assert keep_num_components == self.num_components_kept          # reconsider the timing of LrfSource construction if this assert fails
        return self.rg_info

    def draw(self, cb=None, extra=True, extra_button_specs=None, axes_info=None):
        self.moments_demo_params = None

        def plot_curves(ax, x, y, peaks, baseline, color=None):
            model = self.model
            ax.plot(x, y, color=color, label='data')
            cy_list = []
            for k, params in enumerate(peaks):
                cy = model(x, params)
                cy_list.append(cy)
            props = compute_area_props(cy_list)
            for k, (cy, prop) in enumerate(zip(cy_list, props)):
                ax.plot(x, cy, ":", label='component-%d (%.3g)' % (k, prop))
            ty = np.sum(cy_list, axis=0)
            ax.plot(x, ty, ":", color="red", label='model total')
            ax.plot(x, baseline, color="red", label='baseline')
            ax.legend()

        if axes_info is None:        
            if extra_button_specs is None:
                return_ret_simply = True
                extra_button_specs = [
                                    ("Guess Monopore Params", self.guess_monopore_params)
                                    ("Guess Lnpore Params", self.guess_lnpore_params),
                                    ("Moments Demo", self.moments_demo),
                                    ("Monopore Study", self.monopore_study),
                                    ("Lnpore Study", self.lnpore_study),
                                    ] if extra else []
            else:
                return_ret_simply = False

            with plt.Dp(button_spec=["OK", "Cancel"],
                        extra_button_specs=extra_button_specs):
                fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,5))
                fig.suptitle("LRF source")
                ax1.set_title("UV")
                plot_curves(ax1, self.uv_x, self.uv_y, self.uv_peaks, self.baselines[0], color='blue')
                ax2.set_title("XR")
                plot_curves(ax2, self.xr_x, self.xr_y, self.xr_peaks, self.baselines[1], color='orange')

                if cb is not None:
                    cb(ax2)

                fig.tight_layout()
                ret = plt.show()
        else:
            ax1, ax2 = axes_info
            plot_curves(ax1, self.uv_x, self.uv_y, self.uv_peaks, self.baselines[0], color='blue')
            plot_curves(ax2, self.xr_x, self.xr_y, self.xr_peaks, self.baselines[1], color='orange')
            return

        if return_ret_simply:
            return ret
        else:
            if ret:
                return self.moments_demo_params
            else:
                return
 
    def adapt_params(self, rgs, trs, props, indeces, peak_rgs, params, debug=False):
        # task: clarify the reason for len(rgs) > len(props)
        nc = len(props)
        print("adapt_params nc=", nc, "indeces=", indeces)
        N, T, x0, me, mp, poresize = params[0:NUM_MONOPOREPARAMS]
        scales = params[NUM_MONOPOREPARAMS:]
        ret_scales = np.zeros(nc)
        ret_scales[indeces] = scales
        whole_scale = np.sum(scales)
        for i in sorted(set(np.arange(nc)).difference(set(indeces))):
            ret_scales[i] = props[i]/whole_scale
        ret_params = np.array([x0, poresize, N, me, T, mp]), rgs[0:nc], ret_scales
        return ret_params

    def guess_monopore_params(self, old_style_return=False, debug=False):
        import molass_legacy.Models.Stochastic.MonoporeMoments as module
        reload(module)
        from molass_legacy.Models.Stochastic.MonoporeMoments import study_monopore_moments_impl
        ret = study_monopore_moments_impl(self,
                                          keep_num_components=True,     # this is required for backward compatibility
                                          # trust_max_num=2,              # do not trust Rg's exceeding this number to get possibly better fit
                                          debug=debug)
        if ret is None:
            return
        mnp_params, temp_rgs, unreliable_indeces, params_scaler = ret
        corrected_rgs = temp_rgs
        if len(unreliable_indeces) > 0:
            peak_rgs = self.rg_info[3]
            self.logger.info("peak_rgs=%s are replaced by corrected_rgs=%s with unreliable_indeces=%s", peak_rgs, corrected_rgs, unreliable_indeces)

        if old_style_return:
            N, T, t0, me, mp, poresize = mnp_params[0:6]
            # change the order of parameters to keep backward compatibility
            seccol_params = np.array([t0, poresize, N, me, T, mp])
            print("guess_monopore_params: ", seccol_params, corrected_rgs, mnp_params[6:])
            return seccol_params, corrected_rgs, mnp_params[6:]
        else:
            return mnp_params, corrected_rgs

    def guess_lnpore_params(self, return_rgs=False, use_study=True, use_moments=True, progress_cb=None, debug=False):
        if use_study:
            import molass_legacy.Models.Stochastic.MomentsStudy
            reload(molass_legacy.Models.Stochastic.MomentsStudy)
            from molass_legacy.Models.Stochastic.MomentsStudy import moments_study_impl
            return moments_study_impl(self, return_rgs=True, debug=debug)
        else:
            if debug:
                import molass_legacy.Models.Stochastic.MomentUtils
                reload(molass_legacy.Models.Stochastic.MomentUtils)
            from molass_legacy.Models.Stochastic.MomentUtils import compute_egh_moments
            from molass_legacy.Models.Stochastic.MonoporeGuess import guess_monopore_params_using_moments
            from molass_legacy.Models.Stochastic.LognormalGuess import guess_lognormalpore_params_using_moments
            rgs, trs, orig_props, peak_rgs, peak_trs, props, indeces, qualities = self.compute_rgs(debug=False)
            x, y = self.xr_x, self.xr_y
            model = self.model
            peaks = self.xr_peaks[indeces,:]
            egh_moments_list = compute_egh_moments(peaks)
            if use_moments:
                from molass_legacy.Models.Stochastic.RoughGuess import guess_monopore_params_roughtly
                monopore_params = guess_monopore_params_roughtly(x, y, model, peaks, peak_rgs, props, egh_moments_list)
                self.logger.info("monopore_params in guess_lnpore_params: %s", monopore_params)
                if progress_cb is not None:
                    progress_cb(1)
            else:
                from molass_legacy.Models.Stochastic.SecModelUtils import guess_monopore_params_from_rgdist
                monopore_params = guess_monopore_params_from_rgdist(x, y, peak_rgs, peak_trs, props, debug=False)   
        
            better_monopore_params = guess_monopore_params_using_moments(x, y, egh_moments_list, peak_rgs, props, monopore_params, debug=debug)
            self.logger.info("better_monopore_params in guess_lnpore_params: %s", better_monopore_params)
            if progress_cb is None:
                progress_cb_info = None
            else:
                progress_cb(2)
                progress_cb_info = (2, 10, progress_cb)
            lognormalpore_params = guess_lognormalpore_params_using_moments(x, y, egh_moments_list, peak_rgs, props, better_monopore_params,
                                                                            progress_cb_info=progress_cb_info,
                                                                            debug=debug)
            if progress_cb is not None:
                progress_cb(10)
            if return_rgs:
                return lognormalpore_params, peak_rgs
            else:
                return lognormalpore_params
    
    def moments_demo(self, adapt_backward=True, debug=True):
        if debug:
             import molass_legacy.Models.Stochastic.MomentUtils
             reload(molass_legacy.Models.Stochastic.MomentUtils)
        from molass_legacy.Models.Stochastic.MomentUtils import moments_demo_from_rgdist
        rgs, trs, orig_props, peak_rgs, peak_trs, props, indeces, qualities = self.compute_rgs(debug=False)
        # monopore_params = guess_monopore_params_from_rgdist(self.xr_x, self.xr_y, peak_rgs, peak_trs, props, debug=True)
        monopore_params = None
        demo_ret = moments_demo_from_rgdist(self.xr_x, self.xr_y, self.model, self.xr_peaks[indeces,:], peak_rgs, props, monopore_params,
                                            logger=self.logger, debug=True)
        if demo_ret is not None:
            ret_params = demo_ret[0]
            if adapt_backward:
                ret_params = self.adapt_params(rgs, trs, orig_props, indeces, peak_rgs, ret_params, debug=debug)
            self.moments_demo_params = ret_params

    def monopore_study(self, debug=True):
        if debug:
             import molass_legacy.Models.Stochastic.MonoporeStudy
             reload(molass_legacy.Models.Stochastic.MonoporeStudy)
        from molass_legacy.Models.Stochastic.MonoporeStudy import study
        rgs, trs, orig_props, peak_rgs, peak_trs, props, indeces, qualities = self.compute_rgs(debug=False)
        study(self.xr_x, self.xr_y, self.baselines[1], self.model, self.xr_peaks[indeces,:], peak_rgs, props)

    def lnpore_study(self, debug=True):
        if debug:
             import molass_legacy.Models.Stochastic.LnporeStudy
             reload(molass_legacy.Models.Stochastic.LnporeStudy)
        from molass_legacy.Models.Stochastic.LnporeStudy import study
        rgs, trs, orig_props, peak_rgs, peak_trs, props, indeces, qualities = self.compute_rgs(debug=False)
        study(self.xr_x, self.xr_y, self.baselines[1], self.model, self.xr_peaks[indeces,:], peak_rgs, props)

    def run_simulation_process(self, parent, debug=True):
        import multiprocessing as mp
        from molass_legacy._MOLASS.Processes import register_process
        if debug:
             import molass_legacy.Simulative.SimulationBridge
             reload(molass_legacy.Simulative.SimulationBridge)
        from molass_legacy.Simulative.SimulationBridge import simulation_bridge
        print("run_simulation_process")

        # setting info to pass to the simulation process
        analysis_folder = get_setting('analysis_folder')
        self.run_setting_file = os.path.join(analysis_folder, setting_file) 
        save_settings(self.run_setting_file)

        self.simulation_process = mp.Process(target=simulation_bridge, args=(self, ))
        register_process(self.simulation_process)
        self.simulation_process.start()