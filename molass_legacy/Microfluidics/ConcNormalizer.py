# coding: utf-8
"""
    ConcNormalizer.py

    Copyright (c) 2019, SAXS Team, KEK-PF
"""
import os
import glob
from SerialDataUtils import load_xray_files
from molass_legacy.KekLib.NumpyUtils import np_savetxt_with_comments

def find_conc_file(in_folder):
    files = glob.glob(in_folder + "/Conc*.txt")
    if len(files) > 0:
        _, ret_file = os.path.split(files[0])
    else:
        ret_file = None
    return ret_file

def normalize_impl(in_files, conc_file, out_folder, name_changer, dialog=None):
    conc_data = []
    with open(conc_file) as fh:
        for line in fh.readlines():
            conc_data.append(float(line))

    data_array, comments = load_xray_files(in_files)

    comments.insert(-2, "### comments from concetration normalization begin ###\n")
    comments.insert(-2, "# normalized with %s\n" % conc_file)
    comments.insert(-2, "### comments from concetration normalization end ###\n")

    if dialog is not None:
        dialog.mpb["maximum"] = len(in_files)
        dialog.mpb["value"] = 0

    for k in range(len(in_files)):
        data = data_array[k,:,:]
        data[:,1:] /= conc_data[k]
        _, file = os.path.split(in_files[k])
        file_ = name_changer(file)
        out_file = os.path.join(out_folder, file_)
        # print([k], out_file)
        np_savetxt_with_comments(out_file, data, comments)

        if dialog is not None:
            dialog.num_files_o.set(k+1)
            dialog.mpb["value"] = k+1
            dialog.update()
