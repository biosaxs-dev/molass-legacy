# coding: utf-8
"""
    DataAverager.py

    Copyright (c) 2021, SAXS Team, KEK-PF
"""
import os
import numpy as np
import difflib
from SerialDataUtils import load_xray_files
from molass_legacy.KekLib.NumpyUtils import np_savetxt
from InputSmootherAveraging import IntensitySmootherAveraging
from molass_legacy._MOLASS.SerialSettings import set_setting, get_setting
import molass_legacy.KekLib.CustomMessageBox as MessageBox

def average_impl(num_average, in_files1, out_folder, name_pattern, dialog=None, debug=False):
    num_progress = 0
    num_load_files = 50
    num_average_files = 10

    if dialog is not None:
        dialog.mpb["maximum"] = num_load_files + num_average_files + len(in_files1)
        dialog.mpb["value"] = num_progress

    set_setting('found_lacking_q_values', 0)
    data_array1, comments1, lacking_info1 = load_xray_files(in_files1, return_lacking_info=True)
    if get_setting('found_lacking_q_values'):
        MessageBox.showwarning( "Lacking Q-values",
            "There have been found lacking Q-values\n"
            "in some of the input files.\n"
            "Make sure to confirm them in the molass.log",
            parent=dialog )

    if dialog is not None:
        num_progress += num_load_files
        dialog.mpb["value"] = num_progress
        dialog.update()

    smoother = IntensitySmootherAveraging(data_array1, num_average)
    indeces = np.arange( 0, data_array1.shape[0] )     # these indeces start at 0, not at self.xr_j0
    averaged_array, average_slice_array = smoother( indeces, return_numpy_ndarray=True )

    if dialog is not None:
        num_progress += num_average_files
        dialog.mpb["value"] = num_progress
        dialog.update()

    if debug:
        import molass_legacy.KekLib.DebugPlot as plt
        from MatrixData import simple_plot_3d
        plt.push()
        fig = plt.figure(figsize=(14, 7))
        ax1 = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122, projection='3d')
        simple_plot_3d(ax1, data_array1[:,:,1].T)
        simple_plot_3d(ax2, averaged_array[:,:,1].T)
        plt.show()
        plt.pop()

    out_data_list = []
    for i, data in enumerate( averaged_array ):
        filename = name_pattern % i
        # print( 'saving', filename )
        filepath = os.path.join(out_folder, filename)
        with open( filepath, "wb" ) as fh:
            slice_ = average_slice_array[i]
            fh.write( str.encode( '# Created by averaging the following %d files with []-numbering starting from 1.\n' % ( slice_.stop - slice_.start ) ) )
            for j in range( slice_.start, slice_.stop ):
                fh.write( str.encode( '# [%d] %s\n' % ( j+1, in_files1[j] ) ) )
            fh.write( str.encode( "#\n# Q\tIntensity\tError\n" ) )
            fh.close()
        np_savetxt( filepath, data, mode="a" )
        out_data_list.append(data)
        if dialog is not None:
            n = i+1
            dialog.num_files3.set(n)
            dialog.mpb["value"] = num_progress + n
            dialog.update()

    return out_data_list
