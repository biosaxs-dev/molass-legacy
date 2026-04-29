"""
    Optimizer.OptDataSets.py

    Copyright (c) 2022-2025, SAXS Team, KEK-PF
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

        # If the parent process explicitly exported a LegacyRgCurve (molass-legacy#34 fix),
        # it writes trust.txt into the rg-curve folder to bypass check_rg_folder entirely.
        # This is the primary mechanism (file-based, no SerialSettings dependency).
        # trust_rg_curve_folder in opt_settings.txt is kept as a belt-and-suspenders backup.
        #
        # Additionally, the parent exports to rg_curve_parent/ (a folder the subprocess
        # never writes to), as a robust fallback when rg-curve/ is cleared by an unknown
        # mechanism (molass-legacy#34 ongoing investigation).
        optimizer_folder = os.path.dirname(rg_folder)
        parent_rg_folder = os.path.join(optimizer_folder, "rg_curve_parent")
        _rg_data_files = ['segments.txt', 'qualities.txt', 'slices.txt',
                          'states.txt', 'baseline_type.txt']
        parent_folder_ok = (os.path.exists(parent_rg_folder) and
                            os.path.exists(os.path.join(parent_rg_folder, 'ok.stamp')) and
                            all(os.path.exists(os.path.join(parent_rg_folder, f))
                                for f in _rg_data_files))

        trust_marker = os.path.join(rg_folder, 'trust.txt')
        trust_by_file  = os.path.exists(trust_marker)
        trust_by_setting = get_setting("trust_rg_curve_folder")
        stamp_ok = os.path.exists(rg_folder) and os.path.exists(os.path.join(rg_folder, 'ok.stamp'))
        # Fallback: check that RgCurveProxy's required data files are present
        # even when ok.stamp is missing (e.g. cleared by an unknown mechanism,
        # molass-legacy#34).  This makes the trust check robust to stamp_ok=False.
        data_files_ok = (os.path.exists(rg_folder) and
                         all(os.path.exists(os.path.join(rg_folder, f))
                             for f in _rg_data_files))
        if logger is not None:
            logger.info("rg_folder=%s trust_by_file=%s trust_by_setting=%s "
                        "stamp_ok=%s data_files_ok=%s parent_folder_ok=%s",
                        rg_folder, trust_by_file, trust_by_setting,
                        stamp_ok, data_files_ok, parent_folder_ok)

        # Prefer rg_curve_parent/ when trust is set and it contains valid data
        # (immune to subprocess clearing of rg-curve/).
        if (trust_by_file or trust_by_setting) and parent_folder_ok:
            rg_folder = parent_rg_folder   # redirect to parent-exclusive folder
            rg_folder_ok = True
            if logger is not None:
                logger.info("rg_folder redirected to parent folder: %s", parent_rg_folder)
        elif (trust_by_file or trust_by_setting) and (stamp_ok or data_files_ok):
            rg_folder_ok = True
            if logger is not None:
                logger.info("rg_folder=%s forced OK (trust_by_file=%s, trust_by_setting=%s, "
                            "stamp_ok=%s, data_files_ok=%s)",
                            rg_folder, trust_by_file, trust_by_setting, stamp_ok, data_files_ok)
        else:
            rg_folder_ok = check_rg_folder(rg_folder)
            if logger is not None:
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
                    if logger is not None:
                        logger.info("rg-curve has been copied from %s: check status %s", previous_rg_folder, str(rg_folder_ok))

        if rg_folder_ok:
            rg_curve = RgCurveProxy(xr_curve, rg_folder, progress_cb=progress_cb)
            if trust_by_file or trust_by_setting:
                if logger is not None:
                    logger.info("rg-curve proxy is trusted without trimming consistency check.")
            else:
                xr_restrict_list = get_setting("xr_restrict_list")
                current_rg_trimming = str(None if xr_restrict_list is None else xr_restrict_list[0])
                proxy_rg_trimming = str(rg_curve.rg_trimming)
                if logger is not None:
                    logger.info("trimmings are %s, %s", current_rg_trimming, proxy_rg_trimming)
                if current_rg_trimming == proxy_rg_trimming:
                    if logger is not None:
                        logger.info("rg-curve has been created as a proxy.")
                else:
                    if logger is not None:
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

    # Fix (molass-legacy#34): the legacy loader builds uv_curve.spline with 0-based x,
    # but objective_func evaluates it at original frame positions (uv_x = a*xr_frame + b).
    # When uv_x values exceed the 0-based range, the spline extrapolates flat → wrong uv_y.
    # Rebuild the spline using the stored original frame numbers (uv_curve.x).
    if hasattr(uv_curve, 'spline') and hasattr(uv_curve, 'x'):
        from scipy.interpolate import InterpolatedUnivariateSpline
        _spline_knots = uv_curve.spline.get_knots()
        if abs(float(_spline_knots[0]) - float(uv_curve.x[0])) > 1.0:
            # Spline domain differs from uv_curve.x — rebuild with original frame numbers
            _sy = getattr(uv_curve, 'sy', uv_curve.y)
            uv_curve.spline = InterpolatedUnivariateSpline(uv_curve.x, _sy, ext=3)
            if logger is not None:
                logger.info("uv_curve.spline rebuilt with original frame numbers "
                            "[%.1f, %.1f] (was 0-based [%.1f, %.1f])",
                            float(uv_curve.x[0]), float(uv_curve.x[-1]),
                            float(_spline_knots[0]), float(_spline_knots[-1]))

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
    def __init__(self, sd, corrected_sd, dsets=None, rg_folder=None, E=None,
                 progress_cb=None, compute_rg=False, possibly_relocated=True, current_folder=None):
        self.logger = logging.getLogger(__name__)
        if dsets is None:
            dsets = get_dsets_impl(sd, corrected_sd, progress_cb=progress_cb, rg_folder=rg_folder, logger=self.logger,
                                    compute_rg=compute_rg, possibly_relocated=possibly_relocated, current_folder=current_folder)
        self.dsets = dsets
        D = dsets[0][1]
        if E is None:
            try:                
                E = sd.intensity_array[:,:,2].T
            except:
                D, E, qv, curve = sd.get_xr_data_separate_ly()
        self.E = E

        try:
            self.weight_info = self.compute_weight_info(1/(E + D/100))
        except:
            # LinAlgError("SVD did not converge") as in 20200214_2
            from molass_legacy.KekLib.ExceptionTracebacker import log_exception
            log_exception(self.logger, "compute_weight_info error: ")
            self.weight_info = self.compute_weight_info(1/(E + np.abs(D)/100))
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
    
    def get_x_shifts(self):
        """ Get the x_shifts needed to align the datasets.

        The returned value (x_shifts) can be used as the argument of MplMonitor.run_optimizer()
        and eventually passed to dsets.apply_x_shifts() in the subprocess through "x_shifts.txt"
        text file.

        x_shifts is needed only when the dests has been created in Molass Library, meaning
        that the need arose from the difference with Molass Legacy and Molass Library in handling
        the trimming of the xr-curve and uv-curve.
     
        Parameters
        ----------
        None

        Returns
        -------
        array-like of two elements [xr_shift, uv_shift]
        """
        (xr_curve, D), rg_curve, (uv_curve, U) = self.dsets
        return np.array([xr_curve.x[0], uv_curve.x[0]])

    def apply_x_shifts(self, x_shifts):
        """ Apply the given x_shifts to the datasets.

        Parameters
        ----------
        x_shifts : array-like of two elements [xr_shift, uv_shift]

        Returns
        -------
        None
        """
        (xr_curve, D), rg_curve, (uv_curve, U) = self.dsets
        xr_curve.x += x_shifts[0]
        uv_curve.x += x_shifts[1]
        self.dsets = ((xr_curve, D), rg_curve, (uv_curve, U))
