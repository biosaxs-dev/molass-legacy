# coding: utf-8
"""
    DataSubtractor.py

    Copyright (c) 2019-2021, SAXS Team, KEK-PF
"""
import os
import numpy as np
import difflib
from SerialDataUtils import load_xray_files
from molass_legacy.KekLib.NumpyUtils import np_savetxt_with_comments
from molass_legacy._MOLASS.SerialSettings import set_setting, get_setting
import molass_legacy.KekLib.CustomMessageBox as MessageBox

def subtract_impl(in_files1, in_files2, process_option, out_folder, name_pattern, dialog=None):

    set_setting('found_lacking_q_values', 0)
    data_array1, comments1, lacking_info1 = load_xray_files(in_files1, return_lacking_info=True)
    if get_setting('found_lacking_q_values'):
        MessageBox.showwarning( "Lacking Q-values",
            "There have been found lacking Q-values\n"
            "in some of the input files.\n"
            "Make sure to confirm them in the molass.log",
            parent=dialog )

    if process_option == 0:
        minn, maxn, lacking_list, minn_qv = lacking_info1
        qv = minn_qv if len(lacking_list) > 0 else None
        data_array2, comments2, lacking_info2 = load_xray_files(in_files2, return_lacking_info=True, qv=qv)
        last_line = comments1.pop(-1)
        comments2.pop(-1)

        comments1.append("### comments difference from the subtracted begin ###\n")
        for line in difflib.unified_diff(comments1, comments2):
            if line[0:2] == '+#':
                # print(line)
                comments1.append(line[1:])
        comments1.append("### comments difference from the subtracted end ###\n")
        comments1.append(last_line)
    else:
        average2 = in_files2

    if dialog is not None:
        dialog.mpb["maximum"] = len(in_files1)
        dialog.mpb["value"] = 0

    out_data_list = []
    for k in range(len(in_files1)):
        data1 = data_array1[k,:,:]
        if process_option == 0:
            data2 = data_array2[k,:,:]
        else:
            data2 = average2
        out_data = np.vstack([data1[:,0], data1[:,1] - data2[:,1], np.sqrt(data1[:,2]**2 + data2[:,2]**2)]).T
        filename = name_pattern % k
        np_savetxt_with_comments(os.path.join(out_folder, filename), out_data, comments1)
        out_data_list.append(out_data)

        if dialog is not None:
            dialog.num_files3.set(k+1)
            dialog.mpb["value"] = k+1
            dialog.update()

    return out_data_list
