"""
    DatGnom.py

    Copyright (c) 2022-2023, SAXS Team, KEK-PF
"""
import os
import re
from OurSubprocess import exec_subprocess
from Result import Result
from .AutoRg import autorg_exe_array

DEBUG = False
if DEBUG:
    import logging

class DatgnomExecutor:
    def __init__( self, exe_index=0):
        if DEBUG:
            self.logger = logging.getLogger(__name__)

        self.exe_path   = None
        if len( autorg_exe_array ) > 0:
            # examining datgnom5.exe first comes from molass_legacy.ATSAS 2.7.1
            # TODO: control this after getting the ATSAS version
            for name in [ 'datgnom4.exe', 'datgnom.exe' ]:
                datgnom_path = autorg_exe_array[exe_index].replace( 'autorg.exe', name )
                if os.path.exists( datgnom_path ):
                    self.exe_path   = datgnom_path
                    break

        self.datgnom4_compat = self.exe_path.find("datgnom4") >= 0

    def execute( self, in_file_, rg, out_file_, debug=True ):
        in_file = in_file_.replace('/', '\\')
        out_file = out_file_.replace('/', '\\')
        if self.exe_path is None or rg is None:
            return Result( Qrange=(None, None), Dmax=None, RgPr=None, RgPr_Error=None, IzPr=None, IzPr_Error=None )

        if debug:
            print("self.exe_path=", self.exe_path)

        cmd = [self.exe_path, '-r', '%g' % rg, '-o', out_file, in_file]
        if DEBUG:
            self.logger.info("cmd=%s", str(cmd))
        out, err = exec_subprocess( cmd, shell=False )
        # out = out.replace( '\\', '/' )

        if err != '':
            # print( err )
            # return None
            # atsas-2.8.2 produces warnings
            # TODO: check the exit code instead
            pass

        qrange      = None
        dmax        = None
        rg_pr       = None
        rg_pr_err   = None
        iz_pr       = None
        iz_pr_err   = None

        if self.datgnom4_compat:
            qrange_re = re.compile(r"Angular\s+range\s*:\s+from\s+(\S+)\s+to\s+(\S+)")
            dmax_re = re.compile( r'Dmax\s*=\s+(\d+\.?\d*)')
            rg_re   = re.compile( r'Rg\s+=\s+(\d+\.\S+)\s+\+-\s+(\d+\.\S+)' )
            iz_re   = re.compile( r'I\(0\)\s+=\s+(\d+\.\S+)\s+\+-\s+(\d+\.\S+)' )

            out_lines = out.split( '\n' )
            for line in out_lines:
                m = dmax_re.search( line )
                if m:
                    dmax = float(m.group(1))

            fh = open( out_file )
            for k, line in enumerate(fh):
                if line.find('Angular') >= 0:
                    m = qrange_re.search( line )
                    if m:
                        qrange = [float(m.group(i)) for i in (1, 2)]

                if line.find('Real space:') > 0:
                    # print([k], line)

                    m = rg_re.search( line )
                    if m:
                        rg_pr       = float(m.group(1))
                        rg_pr_err   = float(m.group(2))

                    m = iz_re.search( line )
                    if m:
                        iz_pr       = float(m.group(1))
                        iz_pr_err   = float(m.group(2))
                        break
            fh.close()
        else:
            qrange_re = re.compile(r"Angular\s+range:\s+(\S+)\s+to\s+(\S+)")
            dmax_re = re.compile(r"Real space range:\s+\S+\s+to\s+(\S+)")
            rg_re   = re.compile(r"Real space Rg:\s+(\S+)\s+\+-\s+(\S+)")
            iz_re   = re.compile(r"Real space I\(0\):\s+(\S+)\s+\+-\s+(\S+)")

            fh = open( out_file )
            for k, line in enumerate(fh):
                if line.find('Angular') >= 0:
                    m = qrange_re.search( line )
                    if m:
                        qrange = [float(m.group(i)) for i in (1, 2)]

                if line.find('Real space') > 0:
                    m = dmax_re.search( line )
                    if m:
                        dmax = float(m.group(1))

                    m = rg_re.search( line )
                    if m:
                        rg_pr       = float(m.group(1))
                        rg_pr_err   = float(m.group(2))

                    m = iz_re.search( line )
                    if m:
                        iz_pr       = float(m.group(1))
                        iz_pr_err   = float(m.group(2))
                        break
            fh.close()

        # print( dmax, rg_pr, rg_pr_err, iz_pr, iz_pr_err )

        return Result( Qrange=qrange, Dmax=dmax, RgPr=rg_pr, RgPr_Error=rg_pr_err, IzPr=iz_pr, IzPr_Error=iz_pr_err )

def positive_value( v ):
    v_ = float(v)
    return v_ if v_ > 0 else None

spaces_re = re.compile( r'\s+' )

def datgnom_read_data(datgnom_out_file, null_value=None, null_func=positive_value):

    expr_data_list = []
    real_space_list = []

    expr_data_found     = False
    real_space_found    = False
    fh = open( datgnom_out_file )
    for k, line in enumerate(fh):
        if expr_data_found:
            pass
        else:
            # if line.find( 'Experimental Data and Fit' ) > 0:
            if line.find( 'J EXP' ) > 0:
                expr_data_found = True
                # print( 'expr_data_found' )
            continue

        if real_space_found:
            values = re.split( spaces_re, line )
            if len(values) == 5 and values[1][0].isdigit():
                real_space_list.append( [float(v) for v in values[1:-1] ] )

        else:
            # if line.find( 'Real Space Data' ) > 0:
            if line.find( 'Distance distribution' ) > 0:
                real_space_found = True
                # print( 'real_space_found' )
                continue
            else:
                pass

            values = re.split( spaces_re, line )
            if len(values) == 4 and values[1][0].isdigit():
                # print( line, values )
                expr_data_list.append( [ float(values[1]) ] + [ null_value ]*3 + [ null_func(values[2]) ] )

            elif len(values) == 7 and values[1][0].isdigit():
                # print( line, values )
                expr_data_list.append( [ float(values[1]) ] + [ null_func(v) for v in values[2:-1] ] )

    fh.close()

    return expr_data_list, real_space_list
