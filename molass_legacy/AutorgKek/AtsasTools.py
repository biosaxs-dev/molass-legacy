"""
    AtsasTools.py

    Copyright (c) 2016-2025, SAXS Team, KEK-PF
"""
import os
import numpy as np
from molass_legacy.KekLib.OurSubprocess import exec_subprocess
from molass_legacy.ATSAS.AutoRg import autorg_exe_array
from .IntensityData import IntensityData
from .Guinier import Guinier
from .Result import Result
from .ErrorResult import ErrorResult

def autorg(file, exe_index=0, out_file=None, smaxrg=1.3, verbose=False):
    # print( 'autorg: file=', file )
    assert( os.path.exists( file ) )
    if len(autorg_exe_array) == 0:
        return ErrorResult(), None

    exe_ = autorg_exe_array[ exe_index ]
    opts = ['--smaxrg', str(smaxrg)]

    if out_file is not None:
        opts += ['-o', out_file]

    cmd = [exe_] + opts + [file,  '-f', 'csv']
    out, err = exec_subprocess( cmd, shell=False )
    # print( out, err )
    if err == '':
        lines = out.split( '\n' )
        if len( lines ) >= 3:
            intensity = IntensityData( file )
            x, y, e = intensity.get_guinier_valid_xy()

            File, R, RSD, I, ISD, F, L, Q, A = lines[1].split( ',' )
            if verbose:
                print( R, RSD, I, ISD, F, L, Q, A )
            f, t = int(F)-1, int(L)-1
            f_, t_ = list( map( lambda i: intensity.positive_index[i], ( f, t ) ) )
            Rg = float( R )
            minQ = np.sqrt( x[f_] )
            maxQ = np.sqrt( x[t_] )

            orig_result = Result(
                            type='A',
                            Rg=Rg, Rg_stdev=float( RSD ),
                            I0=float(I), I0_stdev=float(ISD),
                            From=f_, To=t_,
                            Quality=float(Q),
                            Aggregated=float(A),
                            min_qRg=minQ * Rg,
                            max_qRg=maxQ * Rg,
                            min_curvature=None,
                            max_curvature=None,
                            )
            try:
                guinier = Guinier( x, y, e, positive_ratio=intensity.positive_ratio )
                guinier.estimate_rg( [ f_, t_ ], [ 0, 1.3 ] )
                eval_result = guinier.get_result()
            except:
                # as in 20180204/Open01/Open01_00171_sub.dat
                eval_result = None

            return orig_result, eval_result
        else:
            return ErrorResult(), None

    else:
        found = err.find("Data quality")
        if verbose or found < 0:
            print(err)
        return ErrorResult(), None
