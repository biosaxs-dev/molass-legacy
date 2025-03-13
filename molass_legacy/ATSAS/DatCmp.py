"""
    ATSAS.DatCmp.py

    adapted from
        ATSAS.CorMapAnalysis.plot_heatmap(...)

    Copyright (c) 2021, SAXS Team, KEK-PF
"""
import os
import re
import numpy as np
import glob
import subprocess as sp
from .AtsasUtils import get_atsas_bin_path

datcmp_exe = os.path.join(get_atsas_bin_path(), "datcmp.exe")

def run_datcmp(folder=None, files=None, return_dict=False, P_threshold=0.01):
    if files is None:
        assert folder is not None
        files = glob.glob(folder + r"\*")

    ret = sp.run([datcmp_exe] + files, capture_output=True)

    datcmp_data = get_dict_from_log(ret.stdout.decode())

    num_frames = len(files)
    full_data = get_full_data_from_dict(num_frames, datcmp_data, P_threshold=P_threshold)

    if return_dict:
        return full_data, datcmp_data
    else:
        return full_data

def get_dict_from_log(log):
    # borrowed from CorMapAnalysis.py
    data_dict = {"1,2": 0}
    for line in iter(log.splitlines()):
        match_obj = re.match(r'\s* \d{1,} vs', line)
        if match_obj:
            data = line.split()
            if "*" in data[5]:
                data[5] = data[5][:-1]
            data_dict["{},{}".format(data[0], data[2])] = [int(float(data[3])),
                                                           float(data[4]),
                                                           float(data[5])]
    return data_dict

def get_full_data_from_dict(num_frames, datcmp_data, P_threshold=0.01, use_adjP=True):

    if use_adjP:
        P_col = 3
    else:
        P_col = 2

    x_axis = np.linspace(1, num_frames, num_frames)

    full_data = []
    for frame in range(1, num_frames+1):
        pw_data = np.zeros([num_frames, 3])
        for i in range(num_frames):
            if i+1 < frame:
                key = "{},{}".format(i+1, frame)
            elif i+1 > frame:
                key = "{},{}".format(frame, i+1)
            else:
                continue
            pw_data[i, :] = np.asarray(datcmp_data[key])

        pw_data = np.column_stack([x_axis, pw_data])
        good_points = pw_data[pw_data[:, P_col] == 1]
        ok_points = pw_data[np.logical_and(1 > pw_data[:, P_col], pw_data[:, P_col] >= P_threshold)]
        bad_points = pw_data[pw_data[:, P_col] < P_threshold]

        xOrder = list(good_points[:, 0]) + list(ok_points[:, 0]) + list(bad_points[:, 0])
        C_values = list(good_points[:, 1]) + list(ok_points[:, 1]) + list(bad_points[:, 1])
        xData = [-1]*len(good_points[:, 0]) + [0]*len(ok_points[:, 0]) + [1]*len(bad_points[:, 0])

        xOrder_sorted, xData_sorted = (list(t) for t in zip(*sorted(zip(xOrder, xData))))
        full_data.append(xData_sorted)

    ret_array = np.asarray(full_data)
    di = np.diag_indices(num_frames)
    ret_array[di] = 0   # set ok values to the diagonal
    return ret_array

def run_datcmp_from_array(qv, M, E, temp_dir=None, P_threshold=0.01, keep_temp=False):
    import shutil
    if temp_dir is None:
        temp_dir = "datcmp_temp"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)
    for j in range(M.shape[1]):
        file = os.path.join(temp_dir, "data-%04d.dat" % j)
        np.savetxt(file, np.vstack([qv, M[:,j], E[:,j]]).T)
    ret = run_datcmp(temp_dir, P_threshold=P_threshold)
    if not keep_temp:
        shutil.rmtree(temp_dir)
    return ret
