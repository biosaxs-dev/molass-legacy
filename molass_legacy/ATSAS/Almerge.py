"""
    ATSAS/Almerge.py

    Copyright (c) 2016-2023, SAXS Team, KEK-PF
"""
import sys
import os
import glob
import re
import numpy                as np
import subprocess
from OurSubprocess          import exec_subprocess
from molass_legacy._MOLASS.SerialSettings         import get_setting
from Result                 import Result
from molass_legacy.ATSAS.AutoRg import autorg_exe_array

dataline_re = re.compile( r'^\s*\d' )

# generator to read the result file
def read_data_lines( fh ):
    while True:
        try:
            line = fh.readline()
        except:
            # ignore bad data in the tail comment lines from almerge bug
            # UnicodeDecodeError: 'cp932' codec can't decode byte 0xe7 in position 5699: illegal multibyte sequence
            continue

        if not line:
            break
        if dataline_re.match( line ):
            yield line
        else:
            continue

def atsas_file_load( out_file ):
    fh = open( out_file )
    exz_array = np.loadtxt( read_data_lines(fh) )
    fh.close()
    return exz_array

class AlmergeExecutor:
    def __init__( self ):
        self.exe_path   = None
        if len( autorg_exe_array ) > 0:
            almerge_path = autorg_exe_array[0].replace( 'autorg.exe', 'almerge.exe' )
            if os.path.exists( almerge_path ):
                self.exe_path   = almerge_path

        self.out_line_re    = re.compile( r'(\d+)\s*-\s*(\d+)' )

    def execute( self, c_vector, datafiles, indeces, out_file ):
        if self.exe_path is None:
            return None

        self.datafiles  = datafiles
        self.c_vector   = c_vector

        # print( 'indeces=', indeces )
        # print( 'c_vector=', self.c_vector[indeces] )
        cmd = [ self.exe_path ]
        for i in indeces:
            cmd += [ '-c', '%g' % self.c_vector[i], self.datafiles[i] ]

        cmd += [ '-z', '-o', out_file ]

        out, err = exec_subprocess( cmd, shell=False )
        out = out.replace( r'\\', r'/' )

        print( out )
        if err != '':
            print( err )
            return None

        out_lines = out.split( '\n' )
        overlap_from_max = None
        for line in out_lines:
            m = self.out_line_re.search( line )
            if m:
                # print( m.group(1), m.group(2) )
                overlap_from = int( m.group(1) )
                if overlap_from_max is None or overlap_from > overlap_from_max:
                    overlap_from_max = overlap_from

        if not os.path.exists( out_file ):
            # TODO: error info
            return None

        exz_array = atsas_file_load( out_file )

        return Result( exz_array=exz_array, overlap_from_max=overlap_from_max )

    def execute_matrix(self, q, M, E, c_vector, out_file=None):
        from molass_legacy.KekLib.NumpyUtils import np_savetxt

        temp_folder = get_setting('temp_folder')
        datafiles = []
        for k, (data, error)in enumerate(zip(M.T, E.T)):
            file = temp_folder + '/almerge_in_%05d.dat' % k
            datafiles.append(file)
            np_savetxt(file, np.vstack([q, data, error]).T)

        indeces = np.arange(M.shape[1])
        if out_file is None:
            out_file = temp_folder + '/almerge_out.dat'

        return self.execute(c_vector, datafiles, indeces, out_file)
