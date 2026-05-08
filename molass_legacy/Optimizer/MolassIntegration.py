"""
    Optimizer.MolassIntegration.py

    Copyright (c) 2025, SAXS Team, KEK-PF

    Integration with molass-library's data correction pipeline for the optimizer
    subprocess.

    When the subprocess is spawned by molass-library, the raw D/U/E matrices
    derived by the legacy pipeline differ from the parent's because they use
    different baseline correction algorithms.  The .npy override mechanism
    (molass-legacy#38-40) patched these field-by-field after the fact.

    This module (molass-legacy#45, Option B of #44) takes a cleaner approach:
    it re-runs molass-library's own SSD.corrected_copy() pipeline inside the
    subprocess, using the exact trim slices already stored in SerialSettings by
    restore_trimming_info_impl.  The resulting matrices are patched in-place
    into the legacy DataSet objects before OptDataSets is constructed, so the
    downstream optimizer code sees the same data as the parent without needing
    any exported .npy files for D/U/E.

    The .npy overrides in OptDataSets.get_dsets_impl become no-ops (safety net).

    Falls back silently if molass-library is unavailable or a shape mismatch
    occurs, so legacy standalone use is unaffected.
"""
import logging
import numpy as np


def _get_slices_from_serial_settings():
    """Extract (xr_slices, uv_slices) from SerialSettings.

    Must be called AFTER restore_trimming_info_impl has set xr_restrict_list
    and uv_restrict_list from trimming.txt.

    Each list is [TrimmingInfo(j=frames), TrimmingInfo(i=spectral)].
    SsMatrixData.copy(slices=(islice, jslice)) expects (spectral, frame) order.

    Returns
    -------
    (xr_slices, uv_slices) or (None, None) if settings are unavailable.
    """
    from molass_legacy._MOLASS.SerialSettings import get_setting
    xr_restrict = get_setting('xr_restrict_list')
    uv_restrict = get_setting('uv_restrict_list')
    if xr_restrict is None or uv_restrict is None:
        return None, None

    # xr_restrict_list[0] = frame restriction (j-axis)
    # xr_restrict_list[1] = angular/q restriction (i-axis)
    xr_slices = (xr_restrict[1].get_slice(), xr_restrict[0].get_slice())

    # uv_restrict_list[0] = frame restriction (j-axis)
    # uv_restrict_list[1] = wavelength restriction (i-axis)
    uv_slices = (uv_restrict[1].get_slice(), uv_restrict[0].get_slice())

    return xr_slices, uv_slices


def _load_corrected(SSD, in_folder, xr_slices, uv_slices):
    """Load and correct an SSD from in_folder with exact index slices.

    Extracted as a separate function so tests can patch it without needing a
    real data folder on disk.

    Returns the corrected SecSaxsData object.
    """
    import contextlib
    import io
    import warnings

    with contextlib.redirect_stdout(io.StringIO()), \
         contextlib.redirect_stderr(io.StringIO()), \
         warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ssd_raw = SSD(in_folder)
        trimmed = ssd_raw.copy(xr_slices=xr_slices, uv_slices=uv_slices, trimmed=True)
        return trimmed.corrected_copy()


def try_patch_from_molass(in_folder, sd, corrected_sd, logger=None):
    """Patch sd/corrected_sd arrays with molass-library's corrected data.

    Uses SSD(in_folder).copy(slices).corrected_copy() — the same pipeline as
    the parent in-process optimizer — to derive D, E, U matrices.  Patches
    them in-place into the legacy DataSet objects:

      corrected_sd.xr_array[:, :, 1]  ← corrected.xr.M.T   (D matrix)
      corrected_sd.xr_array[:, :, 2]  ← corrected.xr.E.T   (E matrix)
      sd.uv_array                      ← corrected.uv.M      (U matrix)

    Falls back silently (no patch) when molass-library is unavailable, the
    trim slices are missing, or a shape mismatch is detected.

    Parameters
    ----------
    in_folder : str
        Path to the original raw data folder (from in_data_info.txt).
    sd : DataSet
        Legacy trimmed DataSet (sd.uv_array will be patched).
    corrected_sd : DataSet
        Legacy corrected DataSet (xr_array[:,  :, 1/2] will be patched).
    logger : logging.Logger, optional
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    # 1. Check molass-library is importable
    try:
        from molass.DataObjects import SecSaxsData as SSD
    except ImportError:
        logger.info("molass-library not available; skipping molass-library correction patching")
        return

    # 2. Get slice parameters from SerialSettings (set by restore_trimming_info_impl)
    xr_slices, uv_slices = _get_slices_from_serial_settings()
    if xr_slices is None:
        logger.warning("xr_restrict_list not set in SerialSettings; "
                       "skipping molass-library correction patching")
        return

    logger.info("patching sd/corrected_sd from molass-library pipeline "
                "(in_folder=%s, xr_slices=%s, uv_slices=%s)",
                in_folder, str(xr_slices), str(uv_slices))

    try:
        corrected = _load_corrected(SSD, in_folder, xr_slices, uv_slices)

        # --- XR: patch corrected_sd.xr_array[:, :, 1] (D) and [:, :, 2] (E) ---
        xr_M_T = corrected.xr.M.T    # (frames, q)
        expected_xr = corrected_sd.xr_array[:, :, 1].shape
        if xr_M_T.shape == expected_xr:
            corrected_sd.xr_array[:, :, 1] = xr_M_T
            if corrected.xr.E is not None:
                corrected_sd.xr_array[:, :, 2] = corrected.xr.E.T
                logger.info("corrected_sd XR patched (D, E) from molass-library; shape=%s",
                            str(expected_xr))
            else:
                logger.warning("corrected.xr.E is None — E matrix not patched")
                logger.info("corrected_sd XR patched (D only) from molass-library; shape=%s",
                            str(expected_xr))
        else:
            logger.warning("XR shape mismatch: molass M.T %s vs corrected_sd xr_array %s "
                           "— XR patch skipped",
                           str(xr_M_T.shape), str(expected_xr))

        # --- UV: patch sd.uv_array (U) ---
        if corrected.uv is not None:
            uv_M = corrected.uv.M    # (wl, frames)
            if sd.uv_array.shape == uv_M.shape:
                sd.uv_array = uv_M.copy()
                logger.info("sd.uv_array patched (U) from molass-library; shape=%s",
                            str(uv_M.shape))
            else:
                logger.warning("UV shape mismatch: molass uv.M %s vs sd.uv_array %s "
                               "— UV patch skipped",
                               str(uv_M.shape), str(sd.uv_array.shape))
        else:
            logger.info("corrected.uv is None — UV patch skipped")

    except Exception:
        from molass_legacy.KekLib.ExceptionTracebacker import log_exception
        log_exception(logger, "try_patch_from_molass failed: ")
