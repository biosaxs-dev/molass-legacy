"""
Progress chart for optimization GUI.
"""
from molass_legacy.KekLib.NumpyUtils import get_proportional_points
from molass_legacy.KekLib.TimeUtils import friendly_time_str
from molass_legacy.KekLib.ExceptionTracebacker import log_exception
from .FvScoreConverter import convert_score

SHOW_FV_MIN = 0
SHOW_FV_MAX = 100
PROGRESS_X_MARGIN = 1000    # not too large for xmax = 500000
DEVELOP_MODE = True
ENABLE_DEVEL_POPUP_WHILE_BUSY = True

def get_xlim_prog_axes(max_num_evals):
    xmin = -PROGRESS_X_MARGIN
    xmax = max_num_evals + PROGRESS_X_MARGIN
    return xmin, xmax

def get_time_started(fv_array):
    try:
        start_time = fv_array[0, 3]
        time = friendly_time_str(start_time)
    except:
        # IndexError: index 3 is out of bounds for axis 1 with size 2
        time = ""
    return time

def get_time_elapsed(fv_array):
    try:
        start_time = fv_array[0, 3]
        curr_time = fv_array[-1, 3]
        hhmmss = str(curr_time - start_time).split(":")
        time = "%3d.%02d" % tuple([int(s) for s in hhmmss[0:2]])
        # %3d instead of %2d is just for positioning purpose with non-fixed-width fonts.
    except:
        log_exception(None, "get_time_elapsed: ")
        # IndexError: index 3 is out of bounds for axis 1 with size 2
        time = ""
    return time

def guess_ending_time(fv):
    ending_time = ""
    return ending_time

def get_remaining_time(fv):
    remaining_time = ""
    return remaining_time

def draw_progress(self, plot_info):
    for ax in self.prog_axes:
        ax.cla()
    for ax in self.prog_axes[0:3]:
        ax.set_xticklabels([])

    # Function Values
    prog_ax = self.prog_ax

    fv, max_num_evals, x_array = plot_info
    x_, y_ = fv[:,0:2].T
    prog_ax.plot(x_, convert_score(y_))
    prog_ax.set_xlim(-PROGRESS_X_MARGIN, max_num_evals + PROGRESS_X_MARGIN)
    ymin, ymax = prog_ax.get_ylim()
    prog_ax.set_ylim(ymin, ymax)     # these limits will be reset below

    # m = self.best_index
    m = len(x_array) - 1
    if self.curr_index is None:
        self.curr_index = m
    ymin, ymax = prog_ax.get_ylim()
    prog_ax.set_ylim(max(SHOW_FV_MIN, ymin), SHOW_FV_MAX)

    params_type = self.optimizer.params_type

    # Peak Top Positions
    peak_ax = self.prog_axes[1]
    n = self.optimizer.n_components
    x_ = fv[:,0]
    pos_array_list = params_type.get_peak_pos_array_list(x_array)
    for y_ in pos_array_list:
        peak_ax.plot(x_, y_)

    xmin, xmax = self.axes[1].get_xlim()
    xmin_, xmax_ = get_proportional_points(xmin, xmax, [-0.1, 1.1])
    peak_ax.set_ylim(xmin_, xmax_)

    # Rg Values
    rg_ax = self.prog_axes[2]
    rg_start = params_type.get_rg_start_index()
    for k in range(n):
        y_ = x_array[:,rg_start+k]
        rg_ax.plot(x_, y_)
    ymin_rg, ymax_rg = rg_ax.get_ylim()
    rg_ax.set_ylim(10, ymax_rg*1.2)

    # Mapped Range
    map_ax = self.prog_axes[3]
    mr_start = params_type.get_mr_start_index()
    for k in range(2):
        y_ = x_array[:,mr_start+k]
        map_ax.plot(x_, y_)
    map_ax.set_ylim(xmin_, xmax_)

    # Best and Current Result Indicator
    xmin, xmax = get_xlim_prog_axes(max_num_evals)
    best_x = fv[m,0]
    for k, ax in enumerate(self.prog_axes):
        if k > 0:
            ax.set_xlim(xmin, xmax)
            ymin, ymax = ax.get_ylim()
            ax.set_ylim(ymin, ymax)
        ax.plot([best_x, best_x], [ymin, ymax], color='red')
        if self.curr_index != m:
            x = fv[self.curr_index,0]
            ax.plot([x, x], [ymin, ymax], color='yellow')

        if fv.shape[0]-1 not in [m, self.curr_index]:
            x = fv[-1,0]
            ax.plot([x, x], [ymin, ymax], color='gray', alpha=0.3)

    ymin, ymax = map_ax.get_ylim()
    tx = xmax*1.07

    w = 2.2
    ty = ymin*(1-w) + ymax*w
    map_ax.text(tx, ty, "Starting Time", ha="center")

    time_str = get_time_started(fv)
    w = 2.0
    ty = ymin*(1-w) + ymax*w
    map_ax.text(tx, ty, time_str, ha="center", va="center")

    w = 1.6
    ty = ymin*(1-w) + ymax*w
    map_ax.text(tx, ty, "Time Elapsed", ha="center")

    time_str = get_time_elapsed(fv)
    w = 1.4
    ty = ymin*(1-w) + ymax*w
    map_ax.text(tx, ty, time_str, ha="center", va="center")

    w = 0.4
    ty = ymin*(1-w) + ymax*w
    map_ax.text(tx, ty, "Ending Time", ha="center")

    # guess_ending_time() must be called before get_remaining_time()
    time_str = guess_ending_time(fv)
    w = 0.2
    ty = ymin*(1-w) + ymax*w
    map_ax.text(tx, ty, time_str, ha="center", va="center")

    w = 1.0
    ty = ymin*(1-w) + ymax*w
    map_ax.text(tx, ty, "Time Ahead", ha="center")

    time_str = get_remaining_time(fv)
    w = 0.8
    ty = ymin*(1-w) + ymax*w
    map_ax.text(tx, ty, time_str, ha="center", va="center")