"""
    Optimizer.OptDataSets.py

    Copyright (c) 2022-2024, SAXS Team, KEK-PF
"""
import os
import logging
import numpy as np
from molass_legacy._MOLASS.SerialSettings import get_setting, set_setting
from .TheUtils import get_optimizer_folder
from molass_legacy.RgProcess.RgCurve import check_rg_folder, RgCurve
from molass_legacy.RgProcess.RgCurveProxy import RgCurveProxy
import molass_legacy.KekLib.DebugPlot as plt

def get_current_rg_folder(compute_rg=False, possibly_relocated=True, current_folder=None):
    """
    arguments will be used as follows.

    process                  compute_rg  possibly_relocated  current_folder
    ----------------------------------------------------------------------------
    PeakEditor.                 True            False             None
    OptimizerUtils              False           False             None
    FullOptDialog
        (recompute rg-curve)    True            True              specified
    ResultFolderSelector                False           True              specified
    OptimizerMain               False           True              None (getcwd())

    """
    optimizer_folder = get_optimizer_folder()
    rg_folder = os.path.join(optimizer_folder, 'rg-curve')
    if os.path.exists(rg_folder):
        return rg_folder

    if possibly_relocated:
        pass
    else:
        if compute_rg:
            return rg_folder

    relocated_root_folder = rg_folder + "s"
    assert os.path.exists(relocated_root_folder)

    if current_folder is None:
        current_folder = os.getcwd()

    _, jobno = os.path.split(current_folder)
    relocated_folder = os.path.join(relocated_root_folder, jobno)
    if compute_rg:
        pass
    else:
        assert os.path.exists(relocated_folder)

    return relocated_folder

def get_dsets_impl(sd, corrected_sd, progress_cb=None, rg_folder=None, rg_info=True, logger=None,
                    compute_rg=False, possibly_relocated=True, current_folder=None):
    if logger is not None:
        logger = logging.getLogger(__name__)

    D, E, qv, xr_curve = sd.get_xr_data_separate_ly()
    xr_curve_ = None

    if rg_info:
        if rg_folder is None:
            rg_folder = get_current_rg_folder(compute_rg=compute_rg, possibly_relocated=possibly_relocated, current_folder=current_folder)

        rg_folder_ok = check_rg_folder(rg_folder)
        logger.info("rg_folder=%s status is %s", rg_folder, str(rg_folder_ok))

        if not rg_folder_ok:
            previous_rg_folder = get_setting("rg_curve_folder")
            if previous_rg_folder is None:
                # i.e., not ok
                pass
            else:
                if check_rg_folder(previous_rg_folder):
                    from shutil import rmtree, copytree
                    if os.path.exists(rg_folder):
                        # maybe broken
                        rmtree(rg_folder)
                    copytree(previous_rg_folder, rg_folder)
                    rg_folder_ok = check_rg_folder(rg_folder)
                    logger.info("rg-curve has been copied from %s: check status %s", previous_rg_folder, str(rg_folder_ok))

        if rg_folder_ok:
            rg_curve = RgCurveProxy(xr_curve, rg_folder, progress_cb=progress_cb)
            trust_rg_curve_folder = get_setting("trust_rg_curve_folder")
            if trust_rg_curve_folder:
                logger.info("rg-curve proxy is trusted without trimming consistency check.")
            else:
                xr_restrict_list = get_setting("xr_restrict_list")
                current_rg_trimming = str(None if xr_restrict_list is None else xr_restrict_list[0])
                proxy_rg_trimming = str(rg_curve.rg_trimming)
                logger.info("trimmings are %s, %s", current_rg_trimming, proxy_rg_trimming)
                if current_rg_trimming == proxy_rg_trimming:
                    logger.info("rg-curve has been created as a proxy.")
                else:
                    logger.info("the existing rg-curve will be discarded due to trimming inconsistency.")
                    rg_folder_ok = False

        if not rg_folder_ok:
            # create rg_curve from corrected_sd
            D_, E_, qv_, xr_curve_ = corrected_sd.get_xr_data_separate_ly()
            rg_curve = RgCurve(qv_, xr_curve_, D_, E_, progress_cb=progress_cb)

            from molass_legacy.KekLib.BasicUtils import clear_dirs_with_retry
            clear_dirs_with_retry([rg_folder])
            rg_curve.export(rg_folder)
            set_setting("rg_curve_folder", rg_folder)
            logger.info("rg-curve calculation complete.")
    else:
        rg_curve = None

    U, _, wlvec, uv_curve = sd.get_uv_data_separate_ly()

    if False:
        with plt.Dp():
            fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,5))
            fig.suptitle("get_dsets_impl: final")
            ax1.plot(uv_curve.x, uv_curve.y)
            ax2.plot(xr_curve.x, xr_curve.y)
            if xr_curve_ is not None:
                ax2.plot(xr_curve_.x, xr_curve_.y)
            fig.tight_layout()
            plt.show()

    return ((xr_curve, D), rg_curve, (uv_curve, U))

class OptDataSets:
    def __init__(self, sd, corrected_sd, dsets=None, rg_folder=None, progress_cb=None, compute_rg=False, possibly_relocated=True, current_folder=None):
        self.logger = logging.getLogger(__name__)
        if dsets is None:
            dsets = get_dsets_impl(sd, corrected_sd, progress_cb=progress_cb, rg_folder=rg_folder, logger=self.logger,
                                    compute_rg=compute_rg, possibly_relocated=possibly_relocated, current_folder=current_folder)
        self.dsets = dsets
        try:
            D = dsets[0][1]
            E = sd.intensity_array[:,:,2].T
            self.weight_info = self.compute_weight_info(1/(E + D/100))
        except:
            # LinAlgError("SVD did not converge") as in 20200214_2
            from molass_legacy.KekLib.ExceptionTracebacker import log_exception
            log_exception(self.logger, "compute_weight_info error: ")

            D, E, qv, curve = sd.get_xr_data_separate_ly()
            self.weight_info = self.compute_weight_info(1/(E + D/100))  # seems to work for 20200214_2
            self.logger.info("retried with self.compute_weight_info(1/(E + np.abs(D)/100))")

            if False:
                # this plot failes to appear
                import molass_legacy.KekLib.DebugPlot as plt
                from MatrixData import simple_plot_3d
                with plt.Dp():
                    fig, (ax1, ax2)= plt.subplots(ncols=2, figsize=(12,5), subplot_kw=dict(projection="3d"))
                    fig.suptitle("compute_weight_info Error")
                    simple_plot_3d(ax1, D, x=qv)
                    simple_plot_3d(ax2, E, x=qv)
                    fig.tight_layout()
                    plt.show()
        self.E = E

    def relocate_rg_folder(self):
        recompute_rg_curve = get_setting("recompute_rg_curve")
        if recompute_rg_curve:
            import shutil
            from molass_legacy.Optimizer.TheUtils import get_optimizer_folder
            from molass_legacy.KekLib.BasicUtils import clear_dirs_with_retry

            optimizer_folder = get_optimizer_folder()
            rg_folder = os.path.join(optimizer_folder, "rg-curve")
            assert os.path.exists(rg_folder)

            rg_curves_folder = os.path.join(optimizer_folder, "rg-curves")
            clear_dirs_with_retry([rg_curves_folder])
            shutil.move(rg_folder, rg_curves_folder)

            src_path = os.path.join(rg_curves_folder, "rg-curve")
            tgt_path = os.path.join(rg_curves_folder, "000")
            os.rename(src_path, tgt_path)
            self.logger.info("rg-curve has been relocated to %s", tgt_path)

    def compute_weight_info(self, W):
        Wn = W/np.std(W)
        U, s, VT = np.linalg.svd(Wn)
        rank = 1
        s_ = np.sqrt(s[0])
        uw = U[:,0:rank]*s_
        vw = VT[0:rank,:]*s_
        W_ = uw @ vw
        return uw, vw, W_

    def __iter__(self):
        # this is for backward compatibility
        return iter(self.dsets)

    def __getitem__(self, item):
        # this is for backward compatibility
        return self.dsets[item]

    def get_opt_weight_info(self):
        return self.weight_info
