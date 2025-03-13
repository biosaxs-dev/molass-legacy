"""
    AbnormalityCheck.py

    Copyright (c) 2017-2025, SAXS Team, KEK-PF
"""
import numpy as np
from molass_legacy._MOLASS.SerialSettings import get_setting, get_xray_picking
import molass_legacy.KekLib.CustomMessageBox as MessageBox

PLOT_GRADIENT   = True
DEGUB_PLOT      = False

BUBLE_MAX_LENGTH        = 5
ABNORMAL_GRAD_BOUNDARY  = 0.2
TAIL_CHECK_WIDTH        = 10
TAIL_NEG_RATIO          = 0.5

# ------------------------------------------------------------------------------
#   Detect abnormal points caused by bubbles
# ------------------------------------------------------------------------------
def bubble_check(serial_data, **kwargs):
    serial_data.wait_until_ready()
    exclude = []

    if not serial_data.is_serial():
        return exclude

    y   = serial_data.xray_curve.y_orig     # use original considering such a case as 20171226
    # gy  = serial_data.xray_curve.gy
    return bubble_check_impl(y, **kwargs)

BUBBLE_WIDTH_RANGE = (1, 5)
BUBBLE_SEARCH_WIDTH = 10
BUBBLE_GRADIENT_LIMIT = 0.03

def bubble_check_impl(y, debug=False):
    import numpy as np
    from scipy.signal import find_peaks
    max_y = np.max(y)
    height = max_y*0.9
    width = BUBBLE_WIDTH_RANGE
    prominence = height
    peaks, _ = find_peaks(y, height=height, width=width, prominence=prominence)
    if len(peaks) > 0:
        assert len(peaks) == 1
        m = peaks[0]
        start = max(0, m - BUBBLE_SEARCH_WIDTH)
        stop = min(m + BUBBLE_SEARCH_WIDTH, len(y))
        y_ = y[start:stop]
        gy = np.abs(np.gradient(y_))/max_y
        bubbles = start + np.where(gy > BUBBLE_GRADIENT_LIMIT)[0]
    else:
        bubbles = np.array([], dtype=int)
    if debug:
        import matplotlib.pyplot as plt
        x = np.arange(len(y))
        fig, ax = plt.subplots()
        ax.set_title("find_peaks(y, height=%.3g, width=%s, prominence=%.3g)" % (height, width, prominence))
        ax.plot(x, y)
        ax.plot(x[bubbles], y[bubbles], 'o', color='orange', alpha=0.5)
        ax.plot(x[peaks], y[peaks], 'o', color='red', alpha=0.5)
        axt = ax.twinx()
        if False:
            x_ = x[start:stop]
            axt.plot(x_, gy, 'x')
    return bubbles

# ------------------------------------------------------------------------------
#   Exclusion Management
# ------------------------------------------------------------------------------
def exclude_abnormality(serial_data, file_info_table, logger, dialog=None, parent=None, quiet=True):
    if parent is None:
        parent = dialog

    excluded = file_info_table.get_excluded_indeces()
    excluded_spec = None
    if len( excluded ) > 0:
        excluded_ = [ str(i) for i in excluded ]
        reply = MessageBox.askokcancel(
                "Exclude Confirmation",
                'Rows № ' + ','.join( excluded_ )
                            + " are marked to be excluded.\n"
                            + "Those excluded rows will be interpolated instead.\n"
                            + "Ok?",
                parent=parent,
                )
        if not reply:
            return
        excluded_spec = 1
    else:
        ask = True if get_setting('data_exclusion') else False
        exclude = bubble_check(serial_data)
        if len(exclude) > 0:
            excluded_spec = 2
            file_info_table.set_exclude( exclude )
            excluded = exclude

    if len( excluded ) > 0:
        serial_data.exclude_intensities( excluded )
        if excluded_spec == 2:
            if not quiet:
                message = "Xray data at elution points " + str(excluded) + " have been removed and interpolated."
                MessageBox.showinfo(
                    "Exclusion Done", message,
                    parent=parent,
                    )

def update_abnormality_fix_state(sd, file_info_table, logger, dialog=None):
    to_be_excluded = file_info_table.get_excluded_indeces()
    excluded_set = list(sd.excluded_set)
    if excluded_set == to_be_excluded:
        return

    logger.info("updating data exclusion with inconsistent sets: %s vs. %s.", str(to_be_excluded), str(sd.excluded_set))
    sd.exclude_intensities(to_be_excluded)

    if dialog is not None:
        logger.info("updating data exclusion for the original data along with the figures.")
        dialog.serial_data.exclude_intensities(to_be_excluded)
        dialog.fig_frame.draw_figure()
        dialog.update()
