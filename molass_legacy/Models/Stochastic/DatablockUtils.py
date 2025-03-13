"""
    Models.Stochastic.DatablockUtils.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import numpy as np

MIN_QUALITY = 0.3       # to correct under-estimated qualities as in 20210727/data01

def load_datablock_list(moment_rg_file, column_type):
    in_folders = []
    block_list = []
    record_list = None

    def break_procedure():
        nonlocal record_list
        if record_list is None:
            return

        block = np.array(record_list)
        too_low = block[:,3] < MIN_QUALITY
        block[too_low,3] = MIN_QUALITY
        block_list.append(block)
        record_list = None

    with open(moment_rg_file) as fh:
        skip = False
        for k, line in enumerate(fh):
            fields = line[:-1].split(',')
            if fields[1] == '0':
                if fields[3] == column_type:
                    print([k], line[:-1])
                    in_folders.append(fields[0])
                    break_procedure()
                    record_list = []
                    skip = False
                else:
                    skip = True
            else:
                if skip:
                    continue
                print([k], line[:-1])
                record_list.append(np.array([float(v) for v in fields[3:]]))
        break_procedure()

    for k, block in enumerate(block_list):
        print([k], block.shape, block.dtype)

    return in_folders, block_list

def get_block_lrf_src(batch, in_folder):
    from molass_legacy._MOLASS.SerialSettings import clear_temporary_settings, set_setting
    clear_temporary_settings()      # required to avoid
    set_setting('in_folder', in_folder)
    sd = batch.load_data(in_folder)
    batch.prepare(sd, min_num_peaks=2)
    uv_x, uv_y, xr_x, xr_y, baselines = batch.get_curve_xy(return_baselines=True)
    uv_y_ = uv_y - baselines[0]
    xr_y_ = xr_y - baselines[1]
    uv_peaks, xr_peaks = batch.get_modeled_peaks(uv_x, uv_y_, xr_x, xr_y_, affine=True, min_area_prop=None)
    batch.set_lrf_src_args1(uv_x, uv_y, xr_x, xr_y, baselines)  # task: do not require users to do this
    lrf_src = batch.get_lrf_source()
    return lrf_src
()
def get_rg_curves(rg_folders_root, column_type):
    import os
    from RgProcess.RgCurveProxy import RgCurveProxy
    print("get_rg_curves: rg_folders_root=", rg_folders_root)
    mode = 0 if column_type == '1' else 1
    ret_curves = []
    for k, node in enumerate(os.listdir(rg_folders_root)):
        if k % 2 == mode:
            print([k], node)
            fg_folder = os.path.join(rg_folders_root, node)
            rg_curve = RgCurveProxy(None, fg_folder)
            ret_curves.append(rg_curve)
    return ret_curves