"""
    Trimming.MomentTrimming.py

    Copyright (c) 2024-2025, SAXS Team, KEK-PF
"""
from .OptViewRange import OptViewRange
from .TrimmingInfo import TrimmingInfo

def moment_trimming_info_debug_plot(plt, uv_x, uv_y, xr_x, xr_y, ranges_xy, ranges):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,5))
    fig.suptitle('Moment Trimming')
    ax1.plot(uv_x, uv_y)
    # ax1.plot(uv_x, baselines[0])
    ax1.plot(*ranges_xy[0], 'o', color='yellow')
    ax1.axvspan(*ranges[0], alpha=0.2)
    ax2.plot(xr_x, xr_y)
    # ax2.plot(xr_x, baselines[1])
    ax2.plot(*ranges_xy[1], 'o', color='yellow')
    ax2.axvspan(*ranges[1], alpha=0.2)
    fig.tight_layout()

def get_info_from_optview_range(uv_x, uv_y, xr_x, xr_y, num_peaks, debug=False):
    ret_info = []
    if debug:
        ranges = []
        ranges_xy = []

    for x, y in [(uv_x, uv_y), (xr_x, xr_y)]:
        range_ = OptViewRange(x, y, num_peaks, upper=0.05)
        lside, rside = range_.get_range()
        pair = [int(max(0, lside)), int(min(rside, len(x)))]
        ret_info.append(TrimmingInfo(1, *pair, len(x)))

        if debug:
            ranges.append(pair)
            ranges_xy.append((range_.x_, range_.y_))
    
    if debug:
        return ret_info, ranges, ranges_xy
    else:
        return ret_info

def get_moment_trimming_info(sd, debug=False, return_debug_info=False):
    uv_curve = sd.get_uv_curve()
    xr_curve = sd.get_xr_curve()
    uv_x = uv_curve.x
    uv_y = uv_curve.y
    xr_x = xr_curve.x
    xr_y = xr_curve.y

    num_peaks = len(xr_curve.peak_info)
    if debug or return_debug_info:
        ret_info, ranges, ranges_xy = get_info_from_optview_range(uv_x, uv_y, xr_x, xr_y, num_peaks, debug=True)
    else:
        ret_info = get_info_from_optview_range(uv_x, uv_y, xr_x, xr_y, num_peaks)

    if debug:
        import molass_legacy.KekLib.DebugPlot as plt
        with plt.Dp():
            moment_trimming_info_debug_plot(plt, uv_x, uv_y, xr_x, xr_y, ranges_xy, ranges)
            plt.show()

    if return_debug_info:
        return ret_info, uv_x, uv_y, xr_x, xr_y, ranges, ranges_xy
    else:
        return ret_info

def set_moment_trimming_info(sd):
    from molass_legacy._MOLASS.SerialSettings import get_setting, set_setting
    info = get_moment_trimming_info(sd, debug=False)
    uv_restrict_list = get_setting('uv_restrict_list')
    uv_restrict_list[0] = info[0]
    set_setting('uv_restrict_list', uv_restrict_list)
    xr_restrict_list = get_setting('xr_restrict_list')
    xr_restrict_list[0] = info[1]
    set_setting('xr_restrict_list', xr_restrict_list)